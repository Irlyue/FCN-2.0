import time
import argparse
import my_utils as mu
import tensorflow as tf
import utils.image_utils as iu

from post_processing.crf import crf_post_process
from experiment import Experiment
from concurrent import futures
from itertools import repeat
from inputs import voc
from tqdm import tqdm

logger = mu.get_default_logger()
parser = argparse.ArgumentParser()
parser.add_argument('--n_examples', type=int, default=100,
                    help='number of examples to do the grid search')
parser.add_argument('--data', default='/dev/shm/tmp/voc2012_seg_val.tfrecord', type=str)
parser.add_argument('--n_workers', default=4, type=int)


def crf_config_gen():
    config_default = {
        'unary': 'prob',
        'n_steps': 5,
        'gaussian': {
            'compat': 3,
            'sxy': 3
        }

    }
    # bilateral
    # compats = range(4, 5)
    # sxys = range(120, 130, 10)
    # srgbs = range(5, 6)
    compats = range(3, 10)
    sxys = range(30, 150, 5)
    srgbs = range(3, 10)
    for compat, sxy, srgb in ((x, y, z) for x in compats for y in sxys for z in srgbs):
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
    print('Done genreration!')
    time.sleep(10)
    best = {'mIoU': -1}
    with futures.ProcessPoolExecutor(FLAGS.n_workers) as pool:
        for mIoU, config in tqdm(pool.map(eval_crf_config, repeat(data), crf_config_gen())):
            if mIoU > best['mIoU']:
                best.update(config=config, mIoU=mIoU)
                logger.info('Current Best mIoU=%.3f\nconfig=%s', best['mIoU'], best['config'])
    logger.info('Best mIoU=%.3f\nconfig=%s', best['mIoU'], best['config'])


def generate_pairs(tfrecord_path, n_examples):
    logger.info('Generating %s pairs..', n_examples)
    parser = mu.get_default_parser()
    args = parser.parse_args('').__dict__.copy()
    args.update(image_size=(384, 384), aspp_rates=[12], keep_prob=1.0,
                backbone_stride=8, kernel_size=3, data=tfrecord_path,
                gpu_id=0, model_dir='/dev/shm/tmp/fcn')
    config = mu.Config(args)
    exp = Experiment(config, training=False)
    results = []
    meta = voc.load_meta_data('2012')
    ids = voc.load_one_image_set(meta, 'val', task='Segmentation')
    for idx, ends in tqdm(zip(ids[:n_examples], exp.predict())):
        image = iu.read_image(meta.img_path % idx)
        mask = iu.read_mask(meta.cls_img_path % idx)
        probs = ends['up_probs']
        probs = iu.resize_maps(probs, mask.shape)
        results.append((image, mask, probs))
    return results


def eval_crf_config(data, config):
    with mu.MetricMIOU(n_classes=21) as metric:
        images = (item[0] for item in data)
        masks = (item[1] for item in data)
        probs = (item[2] for item in data)
        # for image, mask, prob in data:
            # mask_pred = prob.argmax(axis=-1)
            # metric(mask, mask_pred)
        gen = zip(masks, map(crf_post_process, images, probs, repeat(config)))
        for mask, mask_pred in gen:
            metric(mask, mask_pred)
    return metric.result, config


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS = parser.parse_args()
    grid_search_for_crf()
