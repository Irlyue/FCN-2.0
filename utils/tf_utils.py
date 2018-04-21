import tensorflow as tf
import tensorflow.contrib.slim as slim


def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     activation_fn=tf.nn.relu,
                     use_batch_norm=True):
    """Defines the default ResNet arg scope.

    TODO(gpapan): The batch-normalization related default values above are
      appropriate for use in conjunction with the reference ResNet models
      released at https://github.com/KaimingHe/deep-residual-networks. When
      training ResNets from scratch, they might need to be tuned.

    Args:
      weight_decay: The weight decay to use for regularizing the model.
      batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
      batch_norm_epsilon: Small constant to prevent division by zero when
        normalizing activations by their variance in batch normalization.
      batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
        activations in the batch normalization layer.
      activation_fn: The activation function which is used in ResNet.
      use_batch_norm: Whether or not to use batch normalization.

    Returns:
      An `arg_scope` to use for the resnet models.
    """
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        # 'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'updates_collections': None,
        'fused': None,  # Use fused batch norm if possible.
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=activation_fn,
            normalizer_fn=slim.batch_norm if use_batch_norm else None,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            # The following implies padding='SAME' for pool1, which makes feature
            # alignment easier for dense prediction tasks. This is also used in
            # https://github.com/facebook/fb.resnet.torch. However the accompanying
            # code of 'Deep Residual Learning for Image Recognition' uses
            # padding='VALID' for pool1. You can switch to that choice by setting
            # slim.arg_scope([slim.max_pool2d], padding='VALID').
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


def sparse_softmax_cross_entropy(_sentinel=None, labels=None, logits=None, scope='xentropy', ignore_pixel=255):
    with tf.name_scope(scope):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        mask = tf.not_equal(labels, ignore_pixel)
        wanted = tf.boolean_mask(losses, mask)
        loss = tf.reduce_mean(wanted)
        return loss


def add_moving_average(beta, scope='moving_average'):
    with tf.name_scope(scope):
        variable_averages = tf.train.ExponentialMovingAverage(beta, tf.train.get_or_create_global_step())
        average_op = variable_averages.apply(tf.trainable_variables())
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, average_op)


def get_solver(kind, lr):
    kind = kind.lower()
    if kind == 'adam':
        solver = tf.train.AdamOptimizer(lr)
    elif kind == 'sgd':
        solver = tf.train.GradientDescentOptimizer(lr)
    elif kind == 'momentum':
        solver = tf.train.MomentumOptimizer(lr, momentum=0.9)
    else:
        raise NotImplemented
    return solver


def mean_iou(labels, predictions, num_classes, ignore_pixel=255, scope='miou'):
    """
    :param labels:
    :param predictions:
    :param num_classes: int, including the background class
    :param ignore_pixel: int, default to 255
    :param scope:
    :return:
    """
    with tf.name_scope(scope):
        if ignore_pixel:
            flag = tf.not_equal(labels, ignore_pixel)
            labels = tf.boolean_mask(labels, flag)
            predictions = tf.boolean_mask(predictions, flag)
        return tf.metrics.mean_iou(labels=labels,
                                   predictions=predictions,
                                   num_classes=num_classes)
