import os
import numpy as np
import utils.image_utils as iu


class COCOMapping:
    def __init__(self, coco):
        self._coco = coco
        cat_ids = coco.getCatIds()
        cats = coco.loadCats(cat_ids)
        self.idx2name = {i: item['name'] for i, item in enumerate(cats, start=1)}
        self.idx2name.update({0: 'bg'})
        self.name2idx = {item['name']: i for i, item in enumerate(cats, start=1)}
        self.name2idx.update({'bg': 0})
        self.idx_origin2name = {item['id']: item['name'] for item in cats}
        self.idx_origin2name.update({0: 'bg'})
        self.name2idx_origin = {item['name']: item['id'] for item in cats}
        self.name2idx_origin.update({'bg': 0})
        self.idx_origin2idx = {item['id']: i for i, item in enumerate(cats, start=1)}
        self.idx_origin2idx.update({0: 0})
        self.idx2idx_origin = {i: item['id'] for i, item in enumerate(cats, start=1)}
        self.idx2idx_origin.update({0: 0})
        self.idx2idx_origin_vfunc = np.vectorize(lambda x: self.idx2idx_origin[x])
        self.idx_origin2idx_vfunc = np.vectorize(lambda x: self.idx_origin2idx[x])


class COCOSegWrapper:
    def __init__(self, coco, img_dir=None):
        self._coco = coco
        self.img_dir = img_dir
        self.img_path_fmt = os.path.join(self.img_dir, '%s')
        self.cocomapping = COCOMapping(coco)

    def ann2mask(self, idx):
        """
        Convert COCO annotations to mask.
        :param idx:
        :return: np.array(dtype=uint8)
        """
        img_info = self._coco.loadImgs(ids=idx)[0]
        ann_ids = self._coco.getAnnIds(imgIds=idx, iscrowd=False)
        anns = self._coco.loadAnns(ids=ann_ids)
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            binary_mask = self._coco.annToMask(ann)
            mask[binary_mask == 1] = ann['category_id']
        return mask

    def idx_origin2idx(self, mask):
        return self.cocomapping.idx_origin2idx_vfunc(mask)

    def idx2idx_origin(self, mask):
        return self.cocomapping.idx2idx_origin_vfunc(mask)

    def one_hot2name(self, one_hot):
        indices = np.nonzero(one_hot)[0] + 1
        return [self.cocomapping.idx2name[idx] for idx in indices]

    def iter_image_ids(self):
        """
        Return all the image ids.

        :return: dict_keys
        """
        return self._coco.imgs.keys()

    def read_image(self, idx):
        filename = self.idx2name(idx)
        img = iu.read_image(self.img_path_fmt % filename)
        return img

    def idx2name(self, idx):
        img_info = self._coco.loadImgs(ids=idx)[0]
        return img_info['file_name']

    def __len__(self):
        return len(self._coco.imgs)

    def __iter__(self):
        for idx in self.iter_image_ids():
            try:
                image = self.read_image(idx).astype('uint8')
            except OSError:
                print('Image Info\n', self._coco.loadImgs(idx)[0])
                continue
            if len(image.shape) == 2 or image.shape[2] == 1:
                image = np.stack([image, image, image], axis=2)
            ann = self.ann2mask(idx)
            mask = self.idx_origin2idx(ann).astype('uint8')
            label = self.mask2one_hot(mask).astype('int64')
            yield image, {
                'mask': mask,
                'label': label
            }

    @staticmethod
    def mask2one_hot(mask):
        indices = np.unique(mask) - 1
        indices = [idx for idx in indices if 0 <= idx < 80]
        label = np.zeros(80)
        label[indices] = 1
        return label

    @staticmethod
    def load_colormap(n=255, normalized=True):
        """
        A colormap for displaying mask. Use it like this `plt.imshow(cmap[mask])` and you're
        good to go. Most of the time the default parameters would be ok.

        :param n: int, number of colors in the colormap
        :param normalized: boolean, whether to divide by 255.0 for normalization
        :return:
        """
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((n, 3), dtype=dtype)
        for i in range(n):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap/255 if normalized else cmap
        return cmap