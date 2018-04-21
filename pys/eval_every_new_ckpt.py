import time
import my_utils as mu
import tensorflow as tf

from experiment import Experiment
from models.tf_hooks import EvalBestHook

parser = mu.get_default_parser()

IGNORE_FIRST_CKPT = True


def generate_new_ckpt(model_dir, n_loops=100, wait_secs=60):
    old_ckpts = set()
    for i in range(n_loops):
        ckpt_state = tf.train.get_checkpoint_state(model_dir)
        all_ckpts = set(ckpt_state.all_model_checkpoint_paths) if ckpt_state else set()
        new_ckpts = all_ckpts - old_ckpts
        if len(new_ckpts) == 0:
            print('Wait for %d seconds' % wait_secs)
            try:
                time.sleep(wait_secs)
            except KeyboardInterrupt:
                ans = input('You\'re sure want to exit?(y|n)')
                if ans.startswith('y'):
                    break
        else:
            yield from sorted(new_ckpts, key=lambda x: int(x.split('-')[-1]))
            old_ckpts = all_ckpts


def main():
    experiment = Experiment(config, training=False)

    EvalBestHook.on_start()
    for i, ckpt in enumerate(generate_new_ckpt(config.model_dir)):
        if i == 0 and IGNORE_FIRST_CKPT:
            continue
        experiment.eval(ckpt)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    config = mu.load_config()
    main()
