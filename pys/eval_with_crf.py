import argparse
import numpy as np
import my_utils as mu
import tensorflow as tf

from experiment import Experiment
from models.model import SegInputFunction
from concurrent import futures
from post_processing.crf import crf_post_process
from itertools import repeat

logger = mu.get_default_logger()
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='/tmp/voc2012_seg_val.tfrecord', type=str)


def generate_pairs(tfrecord_path):
    class Resizer:
        def __init__(self):
            pass

        def __enter__(self):
            self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
            self.input_ph = tf.placeholder(tf.float32, shape=[None, None, None])
            self.size_ph = tf.placeholder(tf.int32, shape=[2])
            self.resized = tf.image.resize_images(self.input_ph, self.size_ph)
            return self

        def __call__(self, x, size):
            return self.sess.run(self.resized, feed_dict={self.input_ph: x, self.size_ph: size})

        def __exit__(self, *args):
            self.sess.close()

    def input_batch():
        with tf.Graph().as_default():
            input_fn = SegInputFunction(tfrecord_path, training=False, batch_size=1, n_epochs=1)
            batch_images, batch_labels = input_fn()
            with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
                while True:
                    yield sess.run([batch_images, batch_labels])
    parser = mu.get_default_parser()
    args = parser.parse_args('').__dict__.copy()
    args.update(image_size=(384, 384), aspp_rates=[12], keep_prob=1.0,
                backbone_stride=8, kernel_size=3, data=tfrecord_path,
                gpu_id=0)
    config = mu.Config(args)
    exp = Experiment(config, training=False)
    with Resizer() as resizer:
        for i, (inps, ends) in enumerate(zip(input_batch(), exp.predict())):
            image = inps[0].astype('uint8')[0]
            mask = np.squeeze(inps[1][0]).astype('int64')
            probs = ends['up_probs']
            probs = resizer(probs, mask.shape)
            yield image, mask, probs


class MetricMIOU:
    def __init__(self):
        pass

    def __enter__(self):
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        self.gt_ph = tf.placeholder(tf.int64, shape=[None, None])
        self.pred_ph = tf.placeholder(tf.int64, shape=[None, None])
        self.miou, self.update_op = mu.mean_iou(labels=self.gt_ph, predictions=self.pred_ph, num_classes=21)

    def __call__(self, batch_gt, batch_pred):
        self.sess.run(self.update_op, feed_dict={self.gt_ph: batch_gt, self.pred_ph: batch_pred})

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.result = self.sess.run(self.miou)
        self.sess.close()


def crf_inference(inputs, config):
    image, mask, prob = inputs
    mask_pred = crf_post_process(image, prob, config)
    return mask, mask_pred


def cnn_output(inputs, *args):
    image, mask, prob = inputs
    mask_pred = np.argmax(prob, axis=-1)
    return mask, mask_pred


def eval_crf(config):
    with MetricMIOU as metric:
        with futures.ProcessPoolExecutor as executor:
            for mask, mask_pred in executor.map(crf_inference, generate_pairs(FLAGS.data), repeat(config)):
                metric(mask, mask_pred)
    print('************************', metric.result)


def main():
    config = {
        'unary': 'prob',
        'n_steps': 5,
        'bilateral': {
            'compat': 6,
            'sxy': 100,
            'srgb': 6
        },
        'gaussian': {
            'sxy': 3,
            'compat': 3
        }
    }
    eval_crf(config)


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    main()
