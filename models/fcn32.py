import my_utils as mu
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .model import FCN
from collections import OrderedDict


class FCN32(FCN):
    def __init__(self, backbone):
        super().__init__(backbone, 'fcn32')

    def get_solver(self):
        return tf.train.AdamOptimizer(learning_rate=self.params.lr)

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

        stride = 16
        with tf.variable_scope('up_conv%d' % stride):
            logits_shape = tf.shape(conv6)
            up_logits_shape = [logits_shape[0], logits_shape[1]*stride, logits_shape[2]*stride, logits_shape[3]]
            up_filter = tf.constant(mu.bilinear_upsample_weights(stride, params.n_classes+1), dtype=tf.float32)
            up_logits = tf.nn.conv2d_transpose(conv6, up_filter, output_shape=up_logits_shape,
                                               strides=[1, stride, stride, 1])

        up_output = tf.argmax(up_logits, axis=-1, name='up_output')
        endpoints.update(output=output, up_output=up_output)
        return up_logits, endpoints, backbone_hooks


class VGGFCN32(FCN):
    def __init__(self, backbone):
        super().__init__(backbone, name='VGGFCN32')

    def __call__(self, features, labels, mode, params):
        super().__call__(features, labels, mode, params)

    def get_solver(self):
        return tf.train.AdamOptimizer(learning_rate=self.params.lr)

    def inference(self):
        params = self.params
        endpoints = OrderedDict()
        logits, backbone_hooks = self.backbone(self.features, self.mode, self.params)

        stride = 32
        with tf.variable_scope('up_conv%d' % stride):
            logits_shape = tf.shape(logits)
            up_logits_shape = [logits_shape[0], logits_shape[1]*stride, logits_shape[2]*stride, logits_shape[3]]
            up_filter = tf.constant(mu.bilinear_upsample_weights(stride, params.n_classes+1), dtype=tf.float32)
            up_logits = tf.nn.conv2d_transpose(logits, up_filter, output_shape=up_logits_shape,
                                               strides=[1, stride, stride, 1])
        return up_logits, endpoints, backbone_hooks

