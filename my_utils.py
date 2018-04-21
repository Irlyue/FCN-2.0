from utils.tf_utils import *
from utils.os_utils import *

from configs.configuration import Config
from configs.arguments import get_default_parser


def load_config():
    parser = get_default_parser()
    args = parser.parse_args()
    config = Config(args.__dict__.copy())
    return config


def load_run_config(path=None):
    path = 'configs/rc.json' if path is None else path
    config = load_json_config(path)
    session_config = config.pop('session_config')
    sc = tf.ConfigProto(**session_config)
    config['session_config'] = sc

    run_config = tf.estimator.RunConfig(**config)
    return run_config
