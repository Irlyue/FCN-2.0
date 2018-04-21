import my_utils as mu
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .tf_hooks import RestoreHook
from tensorflow.contrib.framework import assign_from_checkpoint_fn


class FCN:
    def __init__(self, name='fcn'):
        self.name = name

    def __call__(self, features, labels, mode, params):
        self.on_call(features, labels, mode, params)
        self.inference()

    def on_call(self, features, labels, mode, params):
        self.features = features
        self.labels = labels
        self.mode = mode
        self.params = params

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


class BackboneNetwork:
    def __init__(self, name='resnet_v1_101', reg=1e-4, ckpt_path=None):
        self.name = name
        self.reg = reg
        self.ckpt_path = ckpt_path
        resnet = '_'.join(name.split('_')[:-1])
        self.resnet_fn = getattr(mu.load_module('nets.{}'.format(resnet)), name)

    def __call__(self, features, mode, params):
        with slim.arg_scope(mu.resnet_arg_scope(weight_decay=self.reg)):
            with slim.arg_scope([slim.batch_norm], is_training=(mode == tf.estimator.ModeKeys.TRAIN)):
                conv5, _ = self.resnet_fn(features,
                                          num_classes=None,
                                          global_pool=False)
        init_fn = None
        if self.ckpt_path is not None:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            init_fn = assign_from_checkpoint_fn(self.ckpt_path, var_list, ignore_missing_vars=True)
        hooks = [RestoreHook(init_fn)]
        return conv5, hooks
