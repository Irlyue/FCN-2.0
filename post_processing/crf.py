import numpy as np

from pydensecrf.densecrf import DenseCRF2D
from pydensecrf.utils import unary_from_softmax


def crf_post_process(image, probs, config, return_prob=False):
    """
    Perform CRF post process giving the unary.

    :param image: np.array, with shape(height, width, 3)
    :param probs: np.array, same shape as image
    :param config: dict, all the parameters for fully connected CRF post-processing
    :param return_prob: bool, default to False
    :return:
        result: np.array, with shape(height, width) given the segmentation mask.
        prob: np.array, with shape(height, width, n_classes) if `return_prob` is set to True
    """
    height, width, n_classes = probs.shape
    d = DenseCRF2D(width, height, n_classes)

    # unary potential
    assert config['unary'] == 'prob', '<<ERROR>>Only unary potential from probability is supported!'
    unary = unary_from_softmax(probs.transpose((2, 0, 1)))
    d.setUnaryEnergy(unary)

    # pairwise potential
    d.addPairwiseGaussian(**config['gaussian'])
    d.addPairwiseBilateral(rgbim=image, **config['bilateral'])

    # inference
    Q = d.inference(config['n_steps'])
    # result = np.argmax(Q, axis=0).reshape((height, width))
    result = np.array(Q).reshape((height, width, n_classes))
    result = result if return_prob else result.argmax(axis=-1)
    return result


def gen_crf_prob(imgs, probs, config, dtype='float32'):
    result = np.stack((crf_post_process(img, prob, config, True) for img, prob in zip(imgs, probs)), axis=0)
    result = result.astype(dtype)
    return result
