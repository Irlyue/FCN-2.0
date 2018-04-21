import my_utils as mu
import tensorflow as tf

from models.model import BackboneNetwork, InputFunction
from inputs.data_gen import data_gen
from preprocessing.prep import default_prep
from models.fcn32 import FCN32
from models.tf_hooks import RestoreMovingAverageHook

logger = mu.get_default_logger()


class Experiment:
    def __init__(self, config, training):
        self.config = config
        self.training = training
        self.__estimator = None

    def get_input_fn(self, data=None):
        config = self.config
        data = config.data if data is None else data
        #############################
        # data generator function   #
        #############################
        dataset = data.split('-')[0]
        img_set = data.split('-')[1]
        task = data.split('-')[2]
        data_gen_fn = data_gen(dataset, img_set, task, config.image_size)

        #############################
        # image pre-processing      #
        #############################
        image_prep_fn = lambda x: default_prep(x, self.training)

        #############################
        # training input function   #
        #############################
        input_fn = InputFunction(data_gen_fn,
                                 n_classes=config.n_classes,
                                 batch_size=config.batch_size,
                                 n_epochs=config.n_epochs,
                                 prep_fn=image_prep_fn,
                                 shuffle=self.training)
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

    @property
    def estimator(self):
        if self.__estimator is None:
            config = self.config
            backbone = BackboneNetwork(config.backbone, reg=config.reg, ckpt_path=config.ckpt_for_backbone)
            fcn_fn = FCN32(backbone)
            run_config = mu.load_run_config()
            self.__estimator = tf.estimator.Estimator(fcn_fn, model_dir=config.model_dir, params=config, config=run_config)
        return self.__estimator
