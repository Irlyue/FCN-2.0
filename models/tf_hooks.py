import os
import shutil
import my_utils as mu
import tensorflow as tf

logger = mu.get_default_logger()


class RestoreHook(tf.train.SessionRunHook):
    def __init__(self, init_fn):
        self.init_fn = init_fn

    def after_create_session(self, session, coord=None):
        if self.init_fn is not None and session.run(tf.train.get_or_create_global_step()) == 0:
            logger.info('Restoring pre-trained model...')
            self.init_fn(session)


class EvalBestHook(tf.train.SessionRunHook):
    cache_path = '/tmp/best.bin'

    def __init__(self, metric, ckpt_path=None, model_dir=None):
        self.metric = metric
        self.ckpt_path = ckpt_path
        self.model_dir = model_dir
        if ckpt_path and model_dir:
            raise ValueError('Only one of `ckpt_path` or `model_dir` should be provided!')

    @staticmethod
    def on_start():
        """
        Call this method before using this hook
        :return:
        """
        mu.dump_obj((None, -1), EvalBestHook.cache_path)

    def end(self, session):
        current_value = session.run(self.metric)
        best, best_step = mu.load_obj(self.cache_path)
        if best is None or current_value > best:
            best = current_value
            best_step = session.run(tf.train.get_or_create_global_step())
            mu.dump_obj((best, best_step), self.cache_path)
            ckpt_path = self.ckpt_path if self.ckpt_path else os.path.join(self.model_dir, 'model.ckpt-{}'.format(best_step))
            EvalBestHook._save_best(ckpt_path)
        logger.info('Best value {} at step {}, current value {}'.format(best, best_step, current_value))

    @staticmethod
    def _save_best(ckpt_path):
        model_dir, base = os.path.split(ckpt_path)
        for item in [item for item in os.listdir(model_dir) if item.startswith(base)]:
            src_path = os.path.join(model_dir, item)
            dst_path = os.path.join(model_dir, item.replace(base, 'model.ckpt-best'))
            shutil.copy(src_path, dst_path)


class EvalHook(tf.train.SessionRunHook):
    def __init__(self, log_every=100, n_steps=None):
        self.log_every = log_every
        self.n_steps = n_steps
        self.counter = 0

    def after_run(self, run_context, run_values):
        if self.counter % self.log_every == 0:
            logger.info('Evaluation[{}|{}]'.format(self.counter, self.n_steps))
        self.counter += 1


class RestoreMovingAverageHook(tf.train.SessionRunHook):
    def __init__(self, ckpt_dir=None, ckpt_path=None, beta=0.99):
        self.beta = beta
        self.ckpt_dir = ckpt_dir
        self.ckpt_path = ckpt_path
        if (ckpt_dir and ckpt_path) or not (ckpt_dir or ckpt_path):
            raise ValueError('One and only one of the arguments `ckpt_dir` or `ckpt_path` should be provided')
        self.saver = None

    def begin(self):
        with tf.variable_scope('variable_moving_average'):
            variable_averages = tf.train.ExponentialMovingAverage(self.beta)
            self.saver = tf.train.Saver(variable_averages.variables_to_restore())

    def after_create_session(self, session, coord=None):
        ckpt_path = tf.train.latest_checkpoint(self.ckpt_dir) if self.ckpt_path is None else self.ckpt_path
        self.saver.restore(session, ckpt_path)
        logger.info('Model from %s restored', ckpt_path)
