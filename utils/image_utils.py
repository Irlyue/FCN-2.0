import io
import numpy as np

from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

RESIZE_METHOD = {
    'bilinear': Image.BILINEAR,
    'linear': Image.LINEAR,
    'nearest': Image.NEAREST
}


def read_image(path, as_array=True):
    img = Image.open(path)
    if as_array:
        img = np.array(img)
    return img


def read_mask(path, dtype='int64'):
    mask = Image.open(path)
    mask = np.array(mask)
    mask = mask.astype(dtype)
    return mask


def load_color_map_from_image(path):
    img = Image.open(path)
    cmap = np.array(img.getpalette(), dtype=np.uint8).reshape((-1, 3)) / 255.0
    cmap = LinearSegmentedColormap.from_list('', cmap)
    return cmap


def resize_image(img, size, interp='bilinear'):
    assert interp in RESIZE_METHOD, "%s not allowed!" % interp
    # (width, height) --> (height, width)
    size = size[::-1]
    dtype = img.dtype
    interp = RESIZE_METHOD[interp]
    img = Image.fromarray(img)
    img = img.resize(size, resample=interp)
    img = np.array(img, dtype=dtype)
    return img


def resize_mask(mask, size):
    """
    Examples
    --------
    >>> mask = np.random.randint(10, size=(4, 4), dtype='uint8')
    >>> resized = resize_mask(mask, size=(4, 8))
    >>> resized.shape
    (4, 8)
    >>> resized.dtype
    dtype('uint8')
    >>> resized.dtype == mask.dtype
    True
    """
    # (width, height) --> (height, width)
    size = size[::-1]
    mask = Image.fromarray(mask)
    mask = mask.resize(size=size, resample=RESIZE_METHOD['nearest'])
    mask = np.array(mask)
    return mask


def resize_maps(maps, size, normalize=True):
    """
    Resize score maps to some target size.

    Examples
    --------
    >>> prob = np.random.rand(4, 4, 20)
    >>> result = resize_maps(prob, (6, 8))
    >>> result.shape
    (6, 8, 20)
    >>> error = np.abs(result.sum(axis=-1) - 1.0)
    >>> np.all(error < 1e-6)
    True

    :param prob: np.array, with shape like (height, width, n_classes), usually there're
    more than 3 classes.
    :param size: tuple(int, int)
    :param normalize: bool, optional, set it to True if you're resizing probability maps.
    :return:
        prob: np.array,
    """
    maps = np.stack((resize_image(maps[..., i], size) for i in range(maps.shape[-1])), axis=-1)
    if normalize:
        maps /= np.sum(maps, axis=-1, keepdims=True)
    return maps


def save_image(image, path):
    """
    Save image to disk.

    Examples
    --------
    >>> mask = np.random.randint(256, size=(100, 100), dtype='uint8')
    >>> mask_path = '/tmp/mask.png'
    >>> save_image(mask, mask_path)
    >>> loaded = read_mask(mask_path, dtype='uint8')
    >>> loaded.shape == mask.shape
    True
    >>> np.all(mask == loaded)
    True

    :param image:
    :param path:
    :return:
    """
    if type(image) is np.ndarray:
        image = Image.fromarray(image)
    image.save(path)


def decode_raw_png_bytes(contents):
    result = np.array(Image.open(io.BytesIO(contents)))
    return result
