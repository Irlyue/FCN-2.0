import argparse
import numpy as np
import my_utils as mu
import tensorflow as tf

from post_processing.crf import crf_post_process
from experiment import Experiment
from models.model import SegInputFunction
from scipy.misc import imresize
from concurrent import futures
from itertools import repeat

logger = mu.get_default_logger()
parser = argparse.ArgumentParser()
parser.add_argument('--n_examples', type=int, default=100,
                    help='number of examples to do the grid search')
parser.add_argument('--data', default='/tmp/voc2012_seg_val.tfrecord', type=str,
                    help='tfrecord path for prediction')
parser.add_argument('--n_workers', default=5, type=int,
                    help='number of processes to use')


def crf_config_gen():
    config_default = mu.load_json_config('configs/crf_config.json')
    # bilateral
    compats = [3, 4, 5, 6]
    sxys = [30, 40, 50, 60, 70, 80, 90, 100]
    srgbs = [3, 4, 5, 6]
    for compat in compats:
        for sxy in sxys:
            for srgb in srgbs:
                new_params = {
                    'bilateral': {
                        'compat': compat,
                        'sxy': sxy,
                        'srgb': srgb
                    }
                }
                config_new = config_default.copy()
                config_new.update(new_params)
                yield config_new


def grid_search_for_crf():
    data = generate_pairs(FLAGS.data, FLAGS.n_examples)
    best = {'mIoU': -1}
    for mIoU, config in map(eval_crf_config, repeat(data), crf_config_gen()):
        if mIoU > best['mIoU']:
            best.update(config=config, mIoU=mIoU)
            logger.info('Current Best mIoU=%.3f\nconfig=%s', best['mIoU'], best['config'])
    logger.info('Best mIoU=%.3f\nconfig=%s', best['mIoU'], best['config'])


def generate_pairs(tfrecord_path, n_examples):
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
    logger.info('Generating %s pairs..', n_examples)
    parser = mu.get_default_parser()
    args = parser.parse_args('').__dict__.copy()
    args.update(image_size=(384, 384), aspp_rates=[12], keep_prob=1.0,
                backbone_stride=8, kernel_size=3, data=tfrecord_path,
                gpu_id=0)
    config = mu.Config(args)
    exp = Experiment(config, training=False)
    results = []
    with Resizer() as resizer:
        for i, (inps, ends) in enumerate(zip(input_batch(), exp.predict())):
            if i >= n_examples:
                break
            image = inps[0].astype('uint8')[0]
            mask = np.squeeze(inps[1][0]).astype('int64')
            probs = ends['up_probs']
            probs = resizer(probs, mask.shape)
            results.append((image, mask, probs))
    logger.info('Done generation!')
    return results


def eval_crf_config(data_in, config):
    with mu.Timer() as timer:
        with tf.Graph().as_default():
            data = tf.data.Dataset.from_generator(lambda: crf_inference_config(data_in, config),
                                                  output_types=(tf.int64, tf.int64),
                                                  output_shapes=(tf.TensorShape([None, None]),
                                                                 tf.TensorShape([None, None])))
            batch_gt, batch_pred = data.repeat(1).prefetch(FLAGS.n_workers).batch(1).make_one_shot_iterator().get_next()
            mIoU, update_op = mu.mean_iou(labels=batch_gt,
                                          predictions=batch_pred,
                                          num_classes=21)
            with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
                sess.run(tf.local_variables_initializer())
                while True:
                    try:
                        sess.run(update_op)
                    except tf.errors.OutOfRangeError:
                        logger.info('Done!')
                        break
                miou = sess.run(mIoU)
    print('******************************************', timer.eclipsed)
    return miou, config


def crf_inference_config(data, config):
    """
    Do CRF inference using given parameters on input data.

    :param data: tuple[3], (image, mask, probability),
        image: np.array, np.uint8, with shape(height, width, 3)
        mask: np.array, np.int64, with shape(height, width)
        probability: np.array, np.float32, with shape(height, width, n_classes=21)
    :param config:
    :return:
    """
    with futures.ProcessPoolExecutor(FLAGS.n_workers) as pool:
        images = (item[0] for item in data)
        masks = (item[1] for item in data)
        probs = (item[2] for item in data)
        mask_preds = pool.map(crf_post_process, images, probs, repeat(config))
        yield from zip(masks, mask_preds)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS = parser.parse_args()
    grid_search_for_crf()
