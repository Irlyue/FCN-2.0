import my_utils as mu
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .model import FCN
from .optimizer import OptimizerWrapper
from .tf_hooks import EvalBestHook, EvalHook
from collections import OrderedDict


class FCN32(FCN):
    def __init__(self, backbone):
        super().__init__('fcn32')
        self.backbone = backbone

    def __call__(self, features, labels, mode, params):
        with tf.device('/GPU:{}'.format(params.gpu_id)):
            self.on_call(features, labels, mode, params)
            logits, endpoints, backbone_hooks = self.inference()

            if self.predict_mode():
                predictions = {
                    'output': endpoints['output'],
                    'up_output': endpoints['up_output']
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
            data_loss = mu.sparse_softmax_cross_entropy(labels=labels['mask'], logits=logits)
            loss = tf.add_n([reg_loss, data_loss], name='total_loss')
            ##############################
            #          metrics           #
            ##############################
            with tf.name_scope('metrics'):
                with tf.name_scope('accuracy'):
                    accuracy = tf.metrics.accuracy(labels=labels['mask'], predictions=endpoints['up_output'])
                mIoU = mu.mean_iou(labels=labels['mask'], predictions=endpoints['up_output'], num_classes=params.n_classes+1)

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
                solver = OptimizerWrapper(self.params.solver,
                                          params={
                                              self.backbone.name: self.params.lr*self.params.lrp,
                                              'conv6': self.params.lr,
                                              'up_conv32': self.params.lr,
                                          })
                apply_gradient_op = solver.minimize(loss, global_step=self.global_step)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, apply_gradient_op)

                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    train_op = tf.no_op(name='train_op')
                train_hooks = backbone_hooks
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=train_hooks)

    def inference(self):
        params = self.params
        endpoints = OrderedDict()
        conv5, backbone_hooks = self.backbone(self.features, self.mode, self.params)

        conv5 = slim.dropout(conv5, keep_prob=params.keep_prob, is_training=self.train_mode())

        conv6 = slim.conv2d(conv5,
                            num_outputs=params.n_classes+1,
                            kernel_size=1,
                            scope='conv6',
                            activation_fn=None)

        output = tf.argmax(conv6, axis=-1, name='output')

        # up32_conv = slim.conv2d_transpose(conv6,
        #                                   num_outputs=params.n_classes+1,
        #                                   kernel_size=params.backbone_stride*2,
        #                                   stride=params.backbone_stride,
        #                                   scope='up_conv%d' % params.backbone_stride,
        #                                   activation_fn=None)
        up32_conv = tf.image.resize_images(conv6, params.image_size)

        up_output = tf.argmax(up32_conv, axis=-1, name='up_output')
        endpoints.update(output=output, up_output=up_output)
        return up32_conv, endpoints, backbone_hooks

