import my_utils as mu
import tensorflow as tf

from experiment import Experiment

logger = mu.get_default_logger()
parser = mu.get_default_parser()


def main():
    logger.info('\n%s\n', mu.json_out(config.state))
    experiment = Experiment(config, training=False)
    experiment.eval(mu.path_join(config.model_dir, 'model.ckpt-best'))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    config = mu.load_config()
    main()
