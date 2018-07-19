import os
import tempfile
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup


class VOCMeta:
    voc_base_dir = '/home/wenfeng/datasets/VOCdevkit/'
    result_dir = os.path.join(tempfile.gettempdir(), 'voc_results')
    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    idx_to_name = {idx: name for idx, name in enumerate(classes)}
    name_to_idx = {name: idx for idx, name in enumerate(classes)}

    def __init__(self, year='2012'):
        self.base_dir = os.path.join(self.voc_base_dir, 'VOC'+year)
        self.img_dir = os.path.join(self.base_dir, 'JPEGImages')
        self.ann_dir = os.path.join(self.base_dir, 'Annotations')
        self.seg_obj_dir = os.path.join(self.base_dir, 'SegmentationObject')
        self.seg_cls_dir = os.path.join(self.base_dir, 'SegmentationClass')
        self.img_set_dir = os.path.join(self.base_dir, 'ImageSets')
        self.objectness_dir = os.path.join(self.base_dir, 'Objectness')

        # path pattern
        self.img_set_path = os.path.join(self.img_set_dir, '%s/%s.txt')
        self.obj_img_path = os.path.join(self.seg_obj_dir, '%s.png')
        self.cls_img_path = os.path.join(self.seg_cls_dir, '%s.png')
        self.ann_xml_path = os.path.join(self.ann_dir, '%s.xml')
        self.img_path = os.path.join(self.img_dir, '%s.jpg')
        self.objectness_path = os.path.join(self.objectness_dir, '%s.jpg')

        # result direction
        self.seg_obj_res_dir = os.path.join(self.result_dir, 'SegmentationObject')
        self.seg_cls_res_dir = os.path.join(self.result_dir, 'SegmentationClass')
        self.main_res_dir = os.path.join(self.result_dir, 'Main')

    def __repr__(self):
        myself = ''.join('{:<15}: {}\n'.format(key, item) for key, item in self.__dict__.items())
        return myself


class ObjectItem:
    def __init__(self, bs):
        self.name = bs.find('name').decode_contents()
        self.difficult = int(bs.find('difficult').decode_contents())
        self.bbox = tuple(int(item.decode_contents()) for item in
                     (bs.xmin, bs.ymin, bs.xmax, bs.ymax))

    def __repr__(self):
        items = ','.join('{}={}'.format(key, item) for key, item in self.__dict__.items())
        myself = 'Object(%s)' % items
        return myself


class Annotation:
    def __init__(self, bs):
        self.folder = bs.folder.decode_contents()
        self.filename = bs.filename.decode_contents()
        self.database = bs.source.database.decode_contents()
        self.size = tuple(int(item.decode_contents()) for item in
                          (bs.size.height, bs.size.width, bs.size.depth))
        self.segmented = int(bs.segmented.decode_contents())
        objects = []
        for obj in bs.annotation.find_all('object'):
            objects.append(ObjectItem(obj))
        self.num_objects = len(objects)
        self.objects = objects

    def __repr__(self):
        myself = ''.join('{:<15}: {}\n'.format(key, item) for key, item in self.__dict__.items())
        return myself


def load_meta_data(year):
    return VOCMeta(year)


def load_one_image_set(meta, image_set, task='Main'):
    path = meta.img_set_path % (task, image_set)
    with open(path, 'r') as f:
        ids = list(line.strip() for line in f)
    return ids


def load_class_image_set(meta, cls, image_set, task='Main'):
    def line2item(line):
        line = line.strip()
        idx, gt = line.split()
        gt = int(gt)
        return idx, gt
    img_set = cls + '_' + image_set
    path = meta.img_set_path % (task, img_set)
    with open(path, 'r') as f:
        ids = list(line2item(line) for line in f)
    return ids


def load_one_image(meta, index, array=True):
    path = meta.img_path % index
    img = Image.open(path)
    if array:
        img = np.array(img)
    return img


def load_one_obj_image(meta, index, array=True):
    path = meta.obj_img_path % index
    img = Image.open(path)
    if array:
        img = np.array(img)
    return img


def load_one_cls_image(meta, index, array=True):
    path = meta.cls_img_path % index
    img = Image.open(path)
    if array:
        img = np.array(img)
    return img


def load_one_annotation(meta, index):
    path = meta.ann_xml_path % index
    with open(path, 'r') as f:
        xml = ''.join([line.strip('\t') for line in f.readlines()])
        ann = Annotation(BeautifulSoup(xml, 'xml'))
        return ann


def load_color_map(meta, load_from='image'):
    """
    Load the color map. Currently, the color map is loaded from an example image file.
    :param meta: VOCMeta,
    :param load_from: str, where to load the colormap
    :return:
        cmap: np.array, with shape(256, 3) and np.unit8 data type
    """
    if load_from == 'image':
        img = load_one_obj_image(meta, '2007_000032', array=False)
        cmap = np.array(img.getpalette(), dtype=np.uint8).reshape((-1, 3))
    else:
        raise NotImplemented
    return cmap


def load_all_class_label(meta, img_set, task):
    ids = load_one_image_set(meta, img_set, task)
    label_mat = np.zeros((len(ids), 20), dtype=np.float32)
    for cls_name, cls_idx in meta.name_to_idx.items():
        data = load_class_image_set(meta, cls_name, img_set, task)
        img_ids = [item[0] for item in data]
        label = [item[1] for item in data]
        label_mat[:, cls_idx] = label
        assert all(idx1==idx2 for idx1, idx2 in zip(ids, img_ids)), "<<ERROR>> Image id Not match!"

    difficult_mat = np.asarray(label_mat==0, dtype=np.float32)
    label_mat[label_mat == 0] = 1
    label_mat[label_mat == -1] = 0
    labels = {idx: label for idx, label in zip(ids, label_mat)}
    difficults = {idx: diff for idx, diff in zip(ids, difficult_mat)}
    return labels, difficults


def load_one_objectness_image(meta, idx, to_prob=True):
    image = Image.open(meta.objectness_path % idx).convert('L')
    if to_prob:
        image = np.asarray(image, dtype=np.float32)
        prob = image / 255.0
        return prob
    return image


def one_hot_to_class_names(one_hot):
    indices = list(np.nonzero(one_hot)[0])
    return [VOCMeta.idx_to_name[idx] for idx in indices]


def mask_to_one_hot(mask):
    indices = np.unique(mask) - 1
    indices = list(indices[1:-1])
    label = np.zeros((20,))
    label[indices] = 1
    return label
