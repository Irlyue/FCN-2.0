import os
import argparse
import numpy as np
import tensorflow as tf

from inputs import voc
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--year', default='2012', type=str)
parser.add_argument('--image_set', default='train', type=str)
parser.add_argument('--out_dir', default='/tmp', type=str)
parser.add_argument('--output_name', default=None, type=str,
                    help='If not provided, `voc[year]_seg_[image_set].tfrecord` will be used')


def write_to_tfrecord(pairs, filename):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    with tf.python_io.TFRecordWriter(filename) as writer:
        for img_path, ann_path in tqdm(pairs):
            img = np.array(Image.open(img_path), dtype=np.uint8)
            ann = np.array(Image.open(ann_path), dtype=np.uint8)

            height, width = img.shape[:2]
            features = tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'image_raw': _bytes_feature(img.tostring()),
                'mask_raw': _bytes_feature(ann.tostring())
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())


def generate_file_pairs(meta, image_set):
    if image_set == 'trainaug':
        val_ids = voc.load_one_image_set(meta, 'val', task='Segmentation')
        trainval_ids = [item.split('.')[0] for item in os.listdir(os.path.join(meta.base_dir, 'SegmentationClassAug'))]
        trainaug_ids = list(set(trainval_ids) - set(val_ids))
        cls_img_path = os.path.join(meta.base_dir, 'SegmentationClassAug', '%s.png')
        return [(meta.img_path % idx, cls_img_path % idx) for idx in trainaug_ids]
    else:
        ids = voc.load_one_image_set(meta, image_set, task='Segmentation')
        return [(meta.img_path % idx, meta.cls_img_path % idx) for idx in ids]


def convert(image_set, out_path):
    meta = voc.load_meta_data(config.year)
    file_pairs = generate_file_pairs(meta, image_set)
    print('%d file pairs in %s' % (len(file_pairs), image_set))
    write_to_tfrecord(pairs=file_pairs, filename=out_path)
    print('Saved at `%s`' % out_path)


def default_output_name(config):
    return 'voc{}_seg_{}.tfrecord'.format(config.year, config.image_set)


def main():
    output_name = config.output_name or default_output_name(config)
    out_path = os.path.join(config.out_dir, output_name)
    convert(config.image_set, out_path)


if __name__ == '__main__':
    config = parser.parse_args()
    print(config)
    main()