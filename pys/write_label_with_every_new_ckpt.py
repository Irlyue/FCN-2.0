import my_utils as mu
import utils.image_utils as iu
import tensorflow as tf

from experiment import Experiment
from concurrent import futures
from tqdm import tqdm
from post_processing.crf import crf_post_process
from inputs import voc

logger = mu.get_default_logger()
N_WORKERS = 8


class ImageHandler:
    def __init__(self, img_path, mask_path_fmt):
        self.img_path = img_path
        self.mask_save_path = mask_path_fmt

    def __call__(self, idx, ends):
        image = iu.read_image(self.img_path % idx)
        prob = iu.resize_maps(ends['up_probs'], size=image.shape[:2])
        mask_pred = crf_post_process(image, prob, crf_config)
        mask_pred = mask_pred.astype('uint8')
        iu.save_image(mask_pred, self.mask_save_path % idx)


def main():
    logger.info('\n%s\n', mu.json_out(config.state))
    experiment = Experiment(config, training=False)
    meta = voc.load_meta_data('2012')
    ids = voc.load_one_image_set(meta, 'trainaug', 'Segmentation')
    mask_path_fmt = mu.path_join(meta.base_dir, config.mask_dir, '%s.png')

    skipped_ckpts = list(mu.generate_new_ckpt(config.model_dir, 1))
    a, b = skipped_ckpts[-1].split('-')
    b = str(int(b) + 1)
    skipped_ckpts.append(a + '-' + b)
    for i, ckpt in enumerate(mu.generate_new_ckpt(config.model_dir, wait_secs=300)):
        if ckpt in skipped_ckpts:
            continue
        write_to_file(experiment, meta, ids, mask_path_fmt)


def write_to_file(exp, meta, ids, mask_path_fmt):
    handle_one_image = ImageHandler(meta.img_path, mask_path_fmt)
    with futures.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        gen = zip(tqdm(ids), exp.predict())
        for batch in mu.batch_output(gen, N_WORKERS*4):
            id_batch = (item[0] for item in batch)
            ends_batch = (item[1] for item in batch)
            for _ in executor.map(handle_one_image, id_batch, ends_batch):
                pass
    print('Save path: %s' % mask_path_fmt)


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
    main()
