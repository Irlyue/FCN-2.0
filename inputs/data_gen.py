import numpy as np
import tensorflow as tf

from inputs import voc
from scipy.misc import imresize


class DataGen:
    def __init__(self, name='NoName'):
        self.name = name

    def __repr__(self):
        return '{}(n_examples={})'.format(self.name, len(self))

    def __len__(self):
        raise NotImplemented

    def __call__(self, *args, **kwargs):
        yield from self

    def get_shape(self):
        item = self[0]
        return tf.TensorShape(list(item[0].shape)), {key: tf.TensorShape(list(val.shape)) for key, val in item[1].items()}

    def get_dtype(self):
        item = self[0]
        return tf.float32, {key: tf.int32 for key in item[1]}


class VOCDataGen(DataGen):
    def __init__(self, year, img_set, img_size=None):
        super().__init__('VOC%s'%year)
        self.meta = voc.load_meta_data(year)
        self.ids = voc.load_one_image_set(self.meta, img_set, 'Main')
        self.multi_labels, self.difficults = voc.load_all_class_label(self.meta, img_set, 'Main')
        self.img_size = img_size

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return '{}(n_examples={}, image_size={})'.format(self.name, len(self), self.img_size)

    def __getitem__(self, item):
        if type(item) != str:
            item = self.ids[item]
        image = voc.load_one_image(self.meta, item)
        label = self.multi_labels[item]
        difficult = self.difficults[item]
        if self.img_size is not None:
            image = imresize(image, self.img_size)
        return image, {
            'image_origin': image,
            'label': label,
            'difficult': difficult
        }


class VOCSegDataGen(DataGen):
    def __init__(self, year, img_set, img_size=None, background=None):
        super().__init__('VOC%s'%year)
        self.meta = voc.load_meta_data(year)
        self.ids = voc.load_one_image_set(self.meta, img_set, 'Segmentation')
        self.img_set = img_set
        self.img_size= img_size
        self.background = background

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        if type(item) != str:
            item = self.ids[item]
        image = voc.load_one_image(self.meta, item)
        mask = voc.load_one_cls_image(self.meta, item).astype(np.int32)
        label = voc.mask_to_one_hot(mask)
        if self.background:
            mask[mask != 255] -= 1
            mask[mask == -1] = self.background
        if self.img_size is not None:
            image = imresize(image, self.img_size)
            mask = imresize(mask, self.img_size, interp='nearest')
        return image, {
            'image_origin': image,
            'label': label,
            'mask': mask
        }


class VOCSegMainGen(DataGen):
    def __init__(self, img_set, img_size=None):
        super().__init__('VOC2012SegMain')
        self.img_set = img_set
        self.img_size = img_size
        meta = voc.load_meta_data('2012')
        train_main_ids = set(voc.load_one_image_set(meta, 'train', task='Main'))
        val_main_ids = set(voc.load_one_image_set(meta, 'val', task='Main'))
        train_seg_ids = set(voc.load_one_image_set(meta, 'train', task='Segmentation'))
        val_seg_ids = set(voc.load_one_image_set(meta, 'val', task='Segmentation'))

        if img_set == 'train':
            ids = list((train_main_ids | val_main_ids | train_seg_ids) - val_seg_ids)
        elif img_set == 'val':
            ids = list(val_seg_ids)
        else:
            raise NotImplementedError('No such image set as `%s`' % self.img_set)
        self.meta = meta
        self.ids = ids
        self.multi_labels, self.difficults = voc.load_all_class_label(self.meta, 'trainval', 'Main')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        if type(item) != str:
            item = self.ids[item]
        image = voc.load_one_image(self.meta, item)
        image = image if self.img_size is None else imresize(image, self.img_size)

        objectness = voc.load_one_objectness_image(self.meta, item, to_prob=True)
        objectness = objectness if self.img_size is None else imresize(objectness, self.img_size, mode='F')
        # train images may come from either 'Main' or 'Segmentation'
        if self.img_set == 'train':
            # first load from 'Main' task
            try:
                label = self.multi_labels[item]
                diff = self.difficults[item]
                label[diff == 1] = 1
            # once failed, try from 'Segmentation' task
            except:
                mask = voc.load_one_cls_image(self.meta, item).astype(np.int32)
                label = voc.mask_to_one_hot(mask)
            return image, {
                'image_origin': image,
                'label': label,
                'objectness': objectness
            }
        # val images must have segmentation mask
        else:
            mask = voc.load_one_cls_image(self.meta, item).astype(np.int32)
            label = voc.mask_to_one_hot(mask)
            mask = mask if self.img_size is None else imresize(mask, self.img_size, interp='nearest')
            return image, {
                'image_origin': image,
                'label': label,
                'mask': mask,
                'objectness': objectness
            }

    def get_dtype(self):
        labels_type = {'image_origin': tf.int32, 'label': tf.int32, 'objectness': tf.float32}
        if self.img_set != 'train':
            labels_type['mask'] = tf.int32
        return tf.float32, labels_type


dataset_map = {
    'voc2007-Main': lambda img_set, img_size: VOCDataGen('2007', img_set, img_size),
    'voc2012-Main': lambda img_set, img_size: VOCDataGen('2012', img_set, img_size),
    'voc2007-Segmentation': lambda img_set, img_size: VOCSegDataGen('2007', img_set, img_size),
    'voc2012-Segmentation': lambda img_set, img_size: VOCSegDataGen('2012', img_set, img_size),
    'voc2012-SegMain': lambda img_set, img_size: VOCSegMainGen(img_set, img_size)
}


def data_gen(dataset, img_set, task, img_size=None):
    dataset = dataset + '-' + task
    return dataset_map[dataset](img_set, img_size)
