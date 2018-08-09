import my_utils as mu
import tensorflow as tf

from models.model import ResNetBackboneNetwork, SegInputFunction, COCOSegInputFunction
from preprocessing.prep import default_seg_prep
from models.fcn32 import ResNetFCN32, ResNetCrfFCN32
from models.tf_hooks import RestoreMovingAverageHook
from inputs import cocoutils

logger = mu.get_default_logger()


def reshape_and_prep(image, label, training, image_size):
    image = tf.image.resize_images(image, image_size)
    label = tf.image.resize_images(label, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image, label = default_seg_prep(image, label, training)
    label = tf.squeeze(label)
    return image, label


class Experiment:
    def __init__(self, config, training):
        self.config = config
        self.training = training
        self.__estimator = None

    def get_input_fn(self):
        def image_prep_fn(image, label, training):
            return reshape_and_prep(image, label, training, config.image_size)
        config = self.config
        if config.data.endswith('.json'):
            ann_file = config.data
            image_dir = '/'.join(ann_file.split('/')[:-2] + ['images'])
            cw = cocoutils.load_coco(ann_file, image_dir)
            input_fn = COCOSegInputFunction(cw, self.training, batch_size=config.batch_size,
                                            n_epochs=config.n_epochs, prep_fn=image_prep_fn)
        else:
            input_fn = SegInputFunction(config.data, self.training, batch_size=config.batch_size,
                                        n_epochs=config.n_epochs,
                                        prep_fn=image_prep_fn)
        return input_fn

    def train(self, input_fn=None):
        input_fn = self.get_input_fn() if input_fn is None else input_fn
        self.estimator.train(input_fn)

    def eval(self, ckpt_path=None, eval_hooks=None):
        with mu.Timer() as timer:
            hooks = [RestoreMovingAverageHook(ckpt_path=ckpt_path)]
            if eval_hooks:
                hooks.extend(eval_hooks)
            result = self.estimator.evaluate(self.get_input_fn(), checkpoint_path=ckpt_path, hooks=hooks)

        result['data'] = self.config.data
        logger.info('Done in %.fs', timer.eclipsed)
        logger.info('\n%s%s%s\n', '*'*10, result, '*'*10)

    def predict(self, ckpt_path=None, input_fn=None):
        config = self.config
        ckpt_path = ckpt_path or mu.path_join(config.model_dir, 'model.ckpt-best')
        input_fn = input_fn or self.get_input_fn()
        hooks = [RestoreMovingAverageHook(ckpt_path=ckpt_path)]
        return self.estimator.predict(input_fn, checkpoint_path=ckpt_path, hooks=hooks)

    @property
    def estimator(self):
        if self.__estimator is None:
            config = self.config
            # backbone = VGGBackboneNetwork(reg=config.reg, ckpt_path=config.ckpt_for_backbone)
            # fcn_fn = VGGFCN32(backbone)
            backbone = ResNetBackboneNetwork(name=config.backbone,
                                             reg=config.reg,
                                             ckpt_path=config.ckpt_for_backbone,
                                             output_stride=config.backbone_stride)
            fcn_fn = ResNetCrfFCN32(backbone)
            run_config = mu.load_run_config()
            self.__estimator = tf.estimator.Estimator(fcn_fn, model_dir=config.model_dir, params=config, config=run_config)
        return self.__estimator
