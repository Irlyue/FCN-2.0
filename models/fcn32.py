import my_utils as mu
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .model import FCN
from collections import OrderedDict
from .tf_hooks import EvalBestHook, EvalHook
from .model import OptimizerWrapper


class ResNetFCN32(FCN):
    def __init__(self, backbone):
        super().__init__(backbone, 'fcn32')

    def get_solver(self):
        solver = OptimizerWrapper(self.params.solver,
                                  params={
                                      self.backbone.name: self.params.lr*self.params.lrp,
                                      'aspp': self.params.lr
                                  })
        return solver

    def __call__(self, features, labels, mode, params):
        with tf.device('/GPU:{}'.format(params.gpu_id)):
            self.on_call(features, labels, mode, params)
            _, endpoints, backbone_hooks = self.inference()
            if self.predict_mode():
                predictions = {
                    'output': endpoints['output'],
                    'up_output': endpoints['up_output']
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            ##############################
            #          losses            #
            ##############################
            reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
            resized_labels = mu.resize_labels(labels, params.image_size[0] // params.backbone_stride)
            data_loss = mu.sparse_softmax_cross_entropy(labels=resized_labels,
                                                        logits=endpoints['logits'])
            loss = tf.add_n([reg_loss, data_loss], name='total_loss')
            ##############################
            #          metrics           #
            ##############################
            with tf.name_scope('metrics'):
                with tf.name_scope('accuracy'):
                    accuracy = tf.metrics.accuracy(labels=labels, predictions=endpoints['up_output'])
                mIoU = mu.mean_iou(labels=labels, predictions=endpoints['up_output'], num_classes=params.n_classes+1)

            if self.eval_mode():
                metrics = {
                    'accuracy': accuracy,
                    'mIoU': mIoU
                }
                eval_hooks = [EvalHook(), EvalBestHook(metrics['mIoU'][0], model_dir=params.model_dir)]
                return tf.estimator.EstimatorSpec(mode, loss=data_loss, eval_metric_ops=metrics,
                                                  evaluation_hooks=eval_hooks)

            if self.train_mode():
                ##############################
                #          summary           #
                ##############################
                tf.summary.scalar('loss/data_loss', data_loss)
                tf.summary.scalar('loss/reg_loss', reg_loss)
                tf.summary.scalar('lr', params.lr)
                tf.summary.scalar('metrics/accuracy', accuracy[1])
                # tf.summary.scalar('metrics/mIoU', mIoU[1])

                mu.add_moving_average(beta=0.99, scope='variable_moving_average')
                solver = self.get_solver()
                apply_gradient_op = solver.minimize(loss, global_step=self.global_step)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, apply_gradient_op)

                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    train_op = tf.no_op(name='train_op')
                train_hooks = backbone_hooks
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=train_hooks)

    def inference(self):
        def _conv2d(inp, r):
            return slim.conv2d(inp,
                               num_outputs=params.n_classes+1,
                               kernel_size=1 if r == 1 else 3,
                               scope='aspp_{}'.format(r),
                               activation_fn=None,
                               rate=r,
                               padding='SAME')

        params = self.params
        endpoints = OrderedDict()
        conv5, backbone_hooks = self.backbone(self.features, self.mode, self.params)

        conv5 = slim.dropout(conv5, keep_prob=params.keep_prob, is_training=self.train_mode())

        with tf.variable_scope('aspp'):
            paths = []
            for rate in params.aspp_rates:
                paths.append(_conv2d(conv5, rate))
            logits = tf.add_n(paths)

        output = tf.argmax(logits, axis=-1, name='output')

        up_logits = tf.image.resize_bilinear(logits, params.image_size)
        up_output = tf.argmax(up_logits, axis=-1, name='up_output')
        endpoints.update(output=output, up_output=up_output, logits=logits, up_logits=up_logits)
        return up_logits, endpoints, backbone_hooks


class VGGFCN32(FCN):
    def __init__(self, backbone):
        super().__init__(backbone, name='VGGFCN32')

    def get_solver(self):
        return tf.train.AdamOptimizer(learning_rate=self.params.lr)

    def inference(self):
        params = self.params
        endpoints = OrderedDict()
        logits, backbone_hooks = self.backbone(self.features, self.mode, self.params)
        output = tf.argmax(logits, axis=-1)

        stride = 32
        with tf.variable_scope('up_conv%d' % stride):
            logits_shape = tf.shape(logits)
            up_logits_shape = tf.stack([logits_shape[0], logits_shape[1]*stride, logits_shape[2]*stride, logits_shape[3]])
            up_filter = tf.constant(mu.bilinear_upsample_weights(stride, params.n_classes+1), dtype=tf.float32)
            up_logits = tf.nn.conv2d_transpose(logits, up_filter, output_shape=up_logits_shape,
                                               strides=[1, stride, stride, 1])
        up_output = tf.argmax(up_logits, axis=-1)
        endpoints.update(output=output, up_output=up_output)
        return up_logits, endpoints, backbone_hooks

