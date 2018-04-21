import my_utils as mu
import tensorflow as tf


class OptimizerWrapper:
    """
    This is a wrapper class for the tensorflow optimizer. During the network fine-tuning, we would like
    different parts of the network to have different learning rate.
    Usages:
        Case 1: optimizer = OptimizerWrapper('adam', lr=1e-3);
        Case 2: optimizer = OptimizerWrapper('momentum', params={'resnet_v1_101': 1e-1, 'other_layer': 1e-2})
    """
    def __init__(self, type_, lr=1e-3, params=None):
        """
        :param type_: str, choose from{'adam', 'sgd', 'momentum'}
        :param lr: float, optional, used in the Case 1 of the usages.
        :param params: dict, with items(str->float), specify different learning rates for different scopes
        """
        self.params = params
        if params is None:
            self.solver = mu.get_solver(type_, lr)
        else:
            self.vars = []
            self.solvers = []
            for scope, lr in params.items():
                scope_solver = mu.get_solver(type_, lr)
                scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
                self.vars.extend(scope_vars)
                self.solvers.append(scope_solver)

    def minimize(self, loss, global_step=None):
        global_step = tf.train.get_or_create_global_step() if global_step is None else global_step
        if self.params is None:
            return self.solver.minimize(loss, global_step=global_step)
        else:
            counter = 0
            grads = tf.gradients(loss, self.vars)
            ops = []
            for i, (scope, lr) in enumerate(self.params.items()):
                scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
                scope_grads = grads[counter:counter+len(scope_vars)]
                ops.append(self.solvers[i].apply_gradients(zip(scope_grads, scope_vars), global_step=global_step))
                # set it to None so only one solver gets passed in the global step;
                # otherwise, all the solvers will increase the global step
                global_step = None
                counter += len(scope_vars)
            return tf.group(*ops)
