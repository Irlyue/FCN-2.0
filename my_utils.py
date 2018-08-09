import itertools

from time import sleep
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
    for idx, classes in neighbor.items():
        if len(classes) == 1:
            prediction[visited == idx] = list(classes)[0]
    return prediction


def calc_bg_neighbors(prediction):
    """
    Calculate how many neighbors are there for each connected background area. Each connected will be
    labeled with different integer(see the return values for details).

    Examples
    --------
    >>> a = (np.array([
    ...   [0, 0, 0, 0, 0],
    ...   [0, 1, 1, 1, 0],
    ...   [0, 1, 0, 1, 0],
    ...   [0, 1, 1, 1, 0],
    ...   [0, 0, 0, 0, 0]]))
    >>> neighbor, visited = calc_bg_neighbors(a)
    >>> neighbor
    defaultdict(<class 'set'>, {1: {1, -1}, 2: {1}})
    >>> print(visited)
    [[1 1 1 1 1]
     [1 0 0 0 1]
     [1 0 2 0 1]
     [1 0 0 0 1]
     [1 1 1 1 1]]

    :param prediction:
    :return:
        neighbor: dict, integer->set of neighboring classes, image edge is also considered as one class
    and is recognized as -1.
        visited: np.array, same shape as `prediction`, given the connected components.
    """
    def in_range(posi, posj):
        return (0 <= posi < height) and (0 <= posj < width)

    def visit_from(posi, posj):
        tic = time()
        nonlocal counter
        q = deque()
        counter += 1
        q.append((posi, posj))
        visited[posi, posj] = counter
        while len(q) != 0:
            if time() - tic > 5:
                print('<<ERROR>> More than 5 seconds without result')
                neighbor[counter].add(-1)
                neighbor[counter].add(1)
                break
            topi, topj = q.popleft()
            for dx, dy in zip([1, 0, -1, 0], [0, 1, 0, -1]):
                nexti, nextj = topi + dx, topj + dy
                if in_range(nexti, nextj):
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
    return neighbor, visited


def generate_new_ckpt(model_dir, n_loops=None, wait_secs=60):
    old_ckpts = set()
    n_loops = n_loops or int(1e8)
    for _ in itertools.repeat(None, times=n_loops):
        ckpt_state = tf.train.get_checkpoint_state(model_dir)
        all_ckpts = set(ckpt_state.all_model_checkpoint_paths) if ckpt_state else set()
        new_ckpts = all_ckpts - old_ckpts
        if len(new_ckpts) == 0:
            print('Wait for %d seconds' % wait_secs)
            try:
                sleep(wait_secs)
            except KeyboardInterrupt:
                ans = input('Sure you wanna exit?(y|n)')
                if ans.startswith('y'):
                    break
        else:
            yield from sorted(new_ckpts, key=lambda x: int(x.split('-')[-1]))
            old_ckpts = all_ckpts


def batch_output(gen, batch_size):
    """
    Examples
    --------
    >>> list(batch_output(range(5), 2))
    [[0, 1], [2, 3], [4]]
    >>> list(batch_output(range(6), 3))
    [[0, 1, 2], [3, 4, 5]]

    :param gen:
    :param batch_size:
    :return:
    """
    batch_list = []
    counter = 0
    for item in gen:
        batch_list.append(item)
        counter += 1
        if counter == batch_size:
            yield batch_list
            counter = 0
            batch_list = []
    if batch_list:
        yield batch_list
