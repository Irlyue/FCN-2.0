import os
import numpy as np
import my_utils as mu
import tensorflow as tf

from experiment import Experiment
from models.model import SegInputFunction
from concurrent import futures
from post_processing.crf import crf_post_process
from itertools import repeat
from tqdm import tqdm
from inputs import voc

logger = mu.get_default_logger()
parser = mu.get_default_parser()
parser.add_argument('--n_workers', default=8, type=int)
parser.add_argument('--eval_func', default='cnn_output', type=str)


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
        g = tf.Graph()
        with g.as_default():
            input_fn = SegInputFunction(tfrecord_path, training=False, batch_size=1, n_epochs=1)
            batch_images, batch_labels = input_fn()
        with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}), graph=g) as sess:
            while True:
                try:
                    yield sess.run([batch_images, batch_labels])
                except tf.errors.OutOfRangeError:
                    break
    config = mu.Config(FLAGS.__dict__.copy())
    exp = Experiment(config, training=False)
    with Resizer() as resizer:
        for i, (inps, ends) in enumerate(zip(input_batch(), exp.predict())):
            image = inps[0].astype('uint8')[0]
            mask = np.squeeze(inps[1][0]).astype('int64')
            probs = ends['up_probs']
            probs = resizer(probs, mask.shape)
            yield image, mask, probs


def crf_inference_one(item, config):
    image, mask, prob = item
    mask_pred = crf_post_process(image, prob, config)
    return mask, mask_pred


def cnn_output(inputs, *args):
    image, mask, prob = inputs
    mask_pred = np.argmax(prob, axis=-1)
    return mask, mask_pred


def show_class_iou(metric, timer):
    print('********************************************')
    print('{:<15}: {:.0f}s'.format('time', timer.eclipsed))
    print('{:<15}: {:.4f}'.format('mIoU', metric.result))
    class_names = ['background'] + voc.VOCMeta.classes
    for name, iou in zip(class_names, metric.ious):
        print('{:<15}: {:.4f}'.format(name, iou))
    print('********************************************')


def eval_crf(config):
    def batch_output(gen, batch_size):
        batch_list = []
        counter = 0
        for item in gen:
            batch_list.append(item)
            counter += 1
            if counter == batch_size:
                yield batch_list
                counter = 0
                batch_list = []
        if batch_list:
            yield batch_list
    eval_func = globals()[FLAGS.eval_func]
    with futures.ProcessPoolExecutor(FLAGS.n_workers) as executor, mu.MetricMIOU(21) as metric, mu.Timer() as timer:
        try:
            for batches in batch_output(tqdm(generate_pairs(FLAGS.data)), FLAGS.n_workers*4):
                for mask, mask_pred in executor.map(eval_func, batches, repeat(config)):
                    metric(mask, mask_pred)
        except Exception as e:
            print(e)
            pass
    show_class_iou(metric, timer)



def main():
    # config = {
        # 'unary': 'prob',
        # 'n_steps': 5,
        # 'bilateral': {
            # 'compat': 9,
            # 'sxy': 120,
            # 'srgb': 4
        # },
        # 'gaussian': {
            # 'sxy': 3,
            # 'compat': 3
        # }
    # }
    config = {
        'unary': 'prob',
        'n_steps': 5,
        'bilateral': {
            'compat': 4,
            'sxy': 121,
            'srgb': 5
        },
        'gaussian': {
            'sxy': 3,
            'compat': 3
        }
    }
    eval_crf(config)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS = parser.parse_args()
    print(FLAGS)
    main()
