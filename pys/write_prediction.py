import os
import my_utils as mu
import tensorflow as tf
import inputs.voc as voc
import utils.image_utils as iu

from concurrent import futures
from tqdm import tqdm
from experiment import Experiment
from post_processing.crf import crf_post_process

N_WORKERS = 10


def crf_inference_one(inp, config):
    image, prob = inp['image'], inp['prob']
    mask_pred = crf_post_process(image, prob, config)
    return mask_pred


def write_to_file(cofig, crf_config):
    def handle_one_image(idx, ends):
        image_path = meta.img_path % idx
        image = iu.read_image(image_path)
        prob = iu.resize_maps(ends['up_probs'], size=image.shape[:2])
        mask_pred = crf_post_process(image, prob, crf_config)
        iu.save_image(mask_pred, mask_save_path % idx)

    meta = voc.load_meta_data('2012')
    ids = voc.load_one_image_set(meta, 'trainaug', 'Segmentation')
    mask_save_path = os.path.join(meta.base_dir, config.mask_dir, '%s.png')
    mu.create_if_not_exists(os.path.dirname(mask_save_path))
    exp = Experiment(cofig, training=False)
    with futures.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        for idx, ends in tqdm(zip(ids, exp.predict()), total=len(ids)):
            executor.submit(handle_one_image, idx, ends)
    print('Save path: %s' % mask_save_path)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = mu.get_default_parser()
    parser.add_argument('--mask_dir', default='SegmentationClassFCNPred', type=str,
                        help='where to save to mask prediction')
    config = mu.Config(parser.parse_args().__dict__.copy())
    crf_config = {
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
    write_to_file(config, crf_config)
