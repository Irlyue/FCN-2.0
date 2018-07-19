import my_utils as mu
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .optimizer import OptimizerWrapper
from .tf_hooks import EvalBestHook, EvalHook, RestoreHook
from tensorflow.contrib.framework import assign_from_checkpoint_fn


class FCN:
    def __init__(self, backbone, name='fcn'):
        self.backbone = backbone
        self.name = name

    def __call__(self, features, labels, mode, params):
        self.on_call(features, labels, mode, params)
        with tf.device('/GPU:{}'.format(params.gpu_id)):
            logits, endpoints, backbone_hooks = self.inference()
            if self.predict_mode():
                predictions = {
                    'output': endpoints['output'],
                    'up_output': endpoints['up_output']
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
            data_loss = mu.sparse_softmax_cross_entropy(labels=labels, logits=logits)
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

    def on_call(self, features, labels, mode, params):
        self.features = features
        self.labels = labels
        self.mode = mode
        self.params = params

    def get_solver(self):
        solver = OptimizerWrapper(self.params.solver,
                                  params={
                                      self.backbone.name: self.params.lr*self.params.lrp,
                                      'conv6': self.params.lr,
                                      'up_conv32': self.params.lr,
                                  })
        return solver

    def train_mode(self):
        return self.mode == tf.estimator.ModeKeys.TRAIN

    def eval_mode(self):
        return self.mode == tf.estimator.ModeKeys.EVAL

    def predict_mode(self):
        return self.mode == tf.estimator.ModeKeys.PREDICT

    @property
    def global_step(self):
        return tf.train.get_or_create_global_step()

    def inference(self):
        raise NotImplemented


class InputFunction:
    def __init__(self, gen_fn, n_classes, batch_size=32, n_epochs=1, shuffle=False, prep_fn=None, name='NoName'):
        self.gen_fn = gen_fn
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_steps_per_epoch = len(self.gen_fn) // self.batch_size
        self.shuffle = shuffle
        self.prep_fn = prep_fn
        self.img_size = self.gen_fn.img_size
        self.name = name

    def __repr__(self):
        return 'InputFunction(\n{}\n)'.format('\n'.join('\t{}={}'.format(key, value) for key, value in self.__dict__.items()))

    def __call__(self, *args, **kwargs):
        with tf.device('/CPU:31'):
            shapes = self.gen_fn.get_shape()
            dtypes = self.gen_fn.get_dtype()
            print(shapes, dtypes)
            data = tf.data.Dataset.from_generator(self.gen_fn, dtypes, shapes)

            if self.prep_fn:
                data = data.map(lambda a, b: (self.prep_fn(a), b))

            if self.shuffle:
                data = data.shuffle(100)
            data = data.repeat(self.n_epochs)
            # ignoring the last incomplete batch
            data = data.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
            data = data.prefetch(self.batch_size * 8)
            batch_x, batch_y = data.make_one_shot_iterator().get_next()
            return batch_x, batch_y


class SegInputFunction:
    def __init__(self, tfrecord_path, training, batch_size=1, prep_fn=None, n_epochs=1):
        self.tfrecord_path = tfrecord_path
        self.training = training
        self.batch_size = batch_size
        self.prep_fn = prep_fn
        self.n_epochs = n_epochs

    def __call__(self, *args, **kwargs):
        def _parse_function(example_proto):
            features = {
                "height": tf.FixedLenFeature([], tf.int64),
                "width": tf.FixedLenFeature([], tf.int64),
                "image_raw": tf.FixedLenFeature([], tf.string),
                "mask_raw": tf.FixedLenFeature([], tf.string),
                "label": tf.FixedLenFeature([20], tf.int64)
            }
            parsed = tf.parse_single_example(example_proto, features)

            image = tf.decode_raw(parsed['image_raw'], tf.uint8)
            mask = tf.decode_raw(parsed['mask_raw'], tf.uint8)
            height = tf.cast(parsed['height'], tf.int32)
            width = tf.cast(parsed['width'], tf.int32)
            image = tf.reshape(image, shape=(height, width, 3))
            mask = tf.reshape(mask, shape=(height, width, 1))
            label = parsed['label']
            return image, mask, label

        def to_dtype(image, mask, label):
            image = tf.cast(image, tf.float32)
            mask = tf.cast(mask, tf.int32)
            label = tf.cast(label, tf.int32)
            return image, mask, label

        with tf.device('/CPU:0'):
            data = tf.data.TFRecordDataset([self.tfrecord_path])
            data = data.map(_parse_function)
            data = data.map(to_dtype)
            if self.prep_fn:
                data = data.map(lambda image, mask, label: (*self.prep_fn(image, mask, training=self.training), label))

            if self.training:
                data = data.shuffle(100)

            data = data.repeat(self.n_epochs)
            data = data.batch(self.batch_size)
            data = data.prefetch(self.batch_size * 8)
            batch_x, batch_y, batch_z = data.make_one_shot_iterator().get_next()
            return batch_x, {
                'mask': batch_y,
                'label': batch_z
            }


class COCOSegInputFunction:
    def __init__(self, cw, training, batch_size=1, prep_fn=None, n_epochs=1):
        self.cw = cw
        self.training = training
        self.batch_size = batch_size
        self.prep_fn = prep_fn
        self.n_epochs = n_epochs

    def __call__(self, *args, **kwargs):
        with tf.name_scope('InputFunction'):
            output_types = (tf.float32, {'mask': tf.int64, 'label': tf.int64})
            output_shapes = (tf.TensorShape((None, None, 3)),
                             {'mask': tf.TensorShape((None, None)), 'label': tf.TensorShape((80,))})
            data = tf.data.Dataset.from_generator(lambda: iter(self.cw),
                                                  output_types=output_types,
                                                  output_shapes=output_shapes)
            data = data.map(lambda image, labels: (image, labels['mask'][..., None], labels['label']))
            if self.prep_fn:
                data = data.map(lambda image, mask, label: (*self.prep_fn(image, mask, training=self.training), label))
            if self.training:
                data = data.shuffle(100)
            data = data.repeat(self.n_epochs)
            data = data.batch(self.batch_size)
            data = data.prefetch(self.batch_size * 8)
            batch_x, batch_y, batch_z = data.make_one_shot_iterator().get_next()
            return batch_x, {
                'mask': batch_y,
                'label': batch_z
            }


class ResNetBackboneNetwork:
    def __init__(self, name='resnet_v1_101', reg=1e-4, ckpt_path=None, output_stride=16):
        self.name = name
        self.reg = reg
        self.ckpt_path = ckpt_path
        self.output_stride = output_stride
        resnet = '_'.join(name.split('_')[:-1])
        self.resnet_fn = getattr(mu.load_module('nets.{}'.format(resnet)), name)

    def __call__(self, features, mode, params):
        with slim.arg_scope(mu.resnet_arg_scope(weight_decay=self.reg)):
            conv5, _ = self.resnet_fn(features,
                                      is_training=(mode == tf.estimator.ModeKeys.TRAIN),
                                      output_stride=self.output_stride,
                                      num_classes=None,
                                      global_pool=False)
        init_fn = None
        if self.ckpt_path is not None:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            init_fn = assign_from_checkpoint_fn(self.ckpt_path, var_list, ignore_missing_vars=True)
        hooks = [RestoreHook(init_fn)]
        return conv5, hooks


class VGGBackboneNetwork:
    def __init__(self, reg=1e-4, ckpt_path=None):
        self.reg = reg
        self.ckpt_path = ckpt_path
        self.vgg_fn = getattr(mu.load_module('nets.vgg'), 'vgg_16')
        self.name = 'vgg_16'

    def __call__(self, features, mode, params):
        with slim.arg_scope(mu.vgg_arg_scope(weight_decay=self.reg)):
            logits, _ = self.vgg_fn(features,
                                    num_classes=params.n_classes + 1,
                                    is_training=(mode == tf.estimator.ModeKeys.TRAIN),
                                    spatial_squeeze=False,
                                    fc_conv_padding='SAME')
        init_fn = None
        if self.ckpt_path is not None:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            var_list = [item for item in var_list if 'fc8' not in item.name]
            init_fn = assign_from_checkpoint_fn(self.ckpt_path, var_list, ignore_missing_vars=True)
        hooks = [RestoreHook(init_fn)]
        return logits, hooks
