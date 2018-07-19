import os
import my_utils as mu
import tensorflow as tf
import inputs.voc as voc
import utils.image_utils as iu

from concurrent import futures
from tqdm import tqdm
from experiment import Experiment
from post_processing.crf import crf_post_process

N_WORKERS = 4

class ImageHandler:
    def __init__(self, img_path, mask_save_path):
        self.img_path = img_path
        self.mask_save_path = mask_save_path

    def __call__(self, idx, ends):
        image = iu.read_image(self.img_path % idx)
        prob = iu.resize_maps(ends['up_probs'], size=image.shape[:2])
        mask_pred = crf_post_process(image, prob, crf_config)
        mask_pred = mask_pred.astype('uint8')
        iu.save_image(mask_pred, self.mask_save_path % idx)


def crf_inference_one(inp, config):
    image, prob = inp['image'], inp['prob']
    mask_pred = crf_post_process(image, prob, config)
    return mask_pred


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


def write_to_file(cofig, crf_config):
    meta = voc.load_meta_data('2012')
    ids = voc.load_one_image_set(meta, 'trainaug', 'Segmentation')
    mask_save_path = os.path.join(meta.base_dir, config.mask_dir, '%s.png')
    mu.create_if_not_exists(os.path.dirname(mask_save_path))
    exp = Experiment(cofig, training=False)
    handle_one_image = ImageHandler(meta.img_path, mask_save_path)

    with futures.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        gen = zip(tqdm(ids), exp.predict())
        for batch in batch_output(gen, N_WORKERS*4):
            id_batch = (item[0] for item in batch)
            ends_batch = (item[1] for item in batch)
            for _ in executor.map(handle_one_image, id_batch, ends_batch):
                pass
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
