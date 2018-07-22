from sklearn.metrics import confusion_matrix
from skimage import measure
from utils.tf_utils import *
from utils.os_utils import *

from configs.configuration import Config
from configs.arguments import get_default_parser
from collections import deque, defaultdict


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
    args = parser.parse_args().__dict__.copy()
    args['crf_config'] = load_json_config('configs/crf_config.json')
    config = Config(args)
    return config


def load_run_config(path=None):
    path = 'configs/rc.json' if path is None else path
    config = load_json_config(path)
    session_config = config.pop('session_config')
    sc = tf.ConfigProto(**session_config)
    config['session_config'] = sc

    run_config = tf.estimator.RunConfig(**config)
    return run_config


def fill_bg_with_fg(prediction):
    """
    1. get the connected components labeled as background;
    2. test each connected components following:
        -. If component is strictly surrounded(can't be the image edge) by one other class, labeled
        it as that class;
        -. Otherwise, do nothing

    :param prediction:
    :return:
    """
    prediction = prediction.copy()
    neighbor, visited = calc_bg_neighbors(prediction)
    for idx, classes in neighbor:
        if len(classes) == 1:
            prediction[visited == idx] = classes[0]


def calc_bg_neighbors(prediction):
    def inrange(posi, posj):
        return (0 <= posi < height) and (0 <= posj < width)

    def visit_from(posi, posj):
        nonlocal counter
        q = deque()
        counter += 1
        q.append((posi, posj))
        visited[posi, posj] = counter
        while len(q) != 0:
            topi, topj = q.popleft()
            for dx, dy in zip([1, 0, -1, 0], [0, 1, 0, -1]):
                nexti, nextj = topi + dx, topj + dy
                if inrange(nexti, nextj):
                    if prediction[nexti, nextj] == 0 and visited[nexti, nextj] == 0:
                        q.append((nexti, nextj))
                        visited[nexti, nextj] = counter
                    if prediction[nexti, nextj] != 0:
                        neighbor[counter].add(prediction[nexti, nextj])
                else:
                    neighbor[counter].add(-1)
    neighbor = defaultdict(set)
    counter = 0
    height, width = prediction.shape
    visited = np.zeros((height, width), dtype=np.uint8)
    for i, j in ((i, j) for i in range(height) for j in range(width)):
        if visited[i, j] == 0 and prediction[i, j] == 0:
            visit_from(i, j)
            print(i, j)
    return neighbor, visited
