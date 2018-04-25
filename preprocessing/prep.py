import tensorflow as tf

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def default_prep(image, is_training):
    if is_training:
        image = tf.image.random_flip_left_right(image)
    return mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def default_seg_prep(image, label, training):
    if training:
        label = tf.cast(label, tf.float32)
        stacked = tf.stack([image, label], axis=-1)
        stacked = tf.image. random_flip_left_right(stacked)
        image = stacked[:, :, :3]
        label = stacked[:, :, 3:]
        label = tf.cast(label, tf.int32)
    image = mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
    return image, label


def mean_image_subtraction(image, means):
    """Subtracts the given means from each image channel.

    For example:
      means = [123.68, 116.779, 103.939]
      image = _mean_image_subtraction(image, means)

    Note that the rank of `image` must be known.

    Args:
      image: a tensor of size [height, width, C].
      means: a C-vector of values to subtract from each channel.

    Returns:
      the centered image.

    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)
