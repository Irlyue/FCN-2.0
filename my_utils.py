from sklearn.metrics import confusion_matrix
from utils.tf_utils import *
from utils.os_utils import *

from configs.configuration import Config
from configs.arguments import get_default_parser


class MetricMIOU:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __enter__(self):
        self.cm = np.zeros((self.n_classes, self.n_classes), dtype=np.int32)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dias = list(range(self.n_classes))
        count = np.sum(self.cm, axis=0) + np.sum(self.cm, axis=1) - self.cm[dias, dias]
        self.ious = 1.0 * self.cm[dias, dias] / count
        self.result = np.mean(self.ious)

    def __call__(self, ground_truth, prediction):
        assert ground_truth.shape == prediction.shape, '<<ERROR>>Shape not equal!'
        cm = confusion_matrix(ground_truth.flatten(), prediction.flatten(), range(self.n_classes))
        self.cm += cm


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
