import numpy as np

from pydensecrf.densecrf import DenseCRF2D
from pydensecrf.utils import unary_from_softmax


def crf_post_process(image, probs, config):
    """
    Perform CRF post process giving the unary.

    :param image: np.array, with shape(height, width, 3)
    :param probs: np.array, same shape as image
    :param config: dict, all the parameters for fully connected CRF post-processing
    :return:
        result: np.array, with shape(height, width) given the segmentation mask.
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
    result = np.argmax(Q, axis=0).reshape((height, width))
    return result
