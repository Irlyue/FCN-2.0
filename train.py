import my_utils as mu
import tensorflow as tf

from experiment import Experiment

parser = mu.get_default_parser()
logger = mu.get_default_logger()


def main():
    logger.info('\n%s\n', mu.json_out(config.state))
    experiment = Experiment(config, training=True)
    experiment.train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    config = mu.load_config()
    main()
