import time
import itertools
import my_utils as mu
import tensorflow as tf

from experiment import Experiment
from models.tf_hooks import EvalBestHook

logger = mu.get_default_logger()
SKIP_FIRST_N = 3


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
                time.sleep(wait_secs)
            except KeyboardInterrupt:
                ans = input('Sure you wanna to exit?(y|n)')
                if ans.startswith('y'):
                    break
        else:
            yield from sorted(new_ckpts, key=lambda x: int(x.split('-')[-1]))
            old_ckpts = all_ckpts


def main():
    logger.info('\n%s\n', mu.json_out(config.state))
    experiment = Experiment(config, training=False)

    EvalBestHook.on_start()
    for i, ckpt in enumerate(generate_new_ckpt(config.model_dir, wait_secs=300)):
        if i < SKIP_FIRST_N:
            continue
        experiment.eval(ckpt)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    config = mu.load_config()
    main()
