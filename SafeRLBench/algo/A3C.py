"""Asynchronous Actor-Critic Agents."""

from SafeRLBench import AlgorithmBase

try:
    import tensorflow as tf
except:
    from SafeRLBench.error import import_failed
    import_failed('TensorFlow')

import logging

logger = logging.getLogger(__name__)


class A3C(AlgorithmBase):
    """Implementation of the Asynchronous Actor-Critic Agents Algorithm.

    Attributes
    ----------
    """

    def __init__(self, environment, policy, max_it, num_workers=2, rate=0.1):
        """Initialize A3C."""
        if not hasattr(policy, 'sess'):
            raise ValueError('Policy needs `sess` attribute.')

        super(A3C, self).__init__(environment, policy, max_it)

        self.num_workers = num_workers

        # init networks
        with tf.variable_scope('global'):
            self.p_net = _PolicyNet(policy, rate)
            self.v_net = _ValueNet(policy, rate)

        # init advantage op
        # init loss op

    def _initialize(self):
        pass

    def _step(self):
        pass

    def _is_finished(self):
        pass

    def _optimize(self):
        pass


class _Worker(object):
    """Worker thread."""

    def __init__(self, policy, p_net, v_net):
        self.global_policy = policy
        self.global_p_net = p_net
        self.global_v_net = v_net


class _ValueNet(object):
    """Wrapper for the Value function."""

    def __init__(self, policy, rate, train=True):
        with tf.variable_scope('value_estimator'):
            self.X = tf.placeholder(policy.dtype,
                                    shape=policy.X.shape,
                                    name='X')
            self.V = tf.placeholder(policy.dtype,
                                    shape=[None],
                                    name='V')

            self.W = policy.init_weights((policy.layers[0], 1))

            self.V_est = tf.matmul(self.X, self.W)

            if train:
                self.losses = tf.squared_difference(self.V_est, self.V)
                self.loss = tf.reduce_sum(self.losses, name='loss')

                self.opt = tf.train.GradientDescentOptimizer(rate)
                self.grads_and_vars = self.opt.compute_gradients(self.loss)
                self.grads_and_vars = [[[g, v] for g, v in self.grads_and_vars
                                       if g is not None]]
                self.update = self.opt.apply_gradients(self.grads_and_vars)


class _PolicyNet(object):
    """Wrapper for the Policy function."""

    def __init__(self, policy, rate, train=True):
        with tf.variable_scope('policy_estimator'):

            self.X = policy.X
            self.y = policy.y

            self.W = policy.W

            self.y_pred = policy.y_pred

            if train:
                self.losses = tf.squared_difference(self.y_pred, self.y)
                self.loss = tf.reduce_sum(self.losses, name='loss')

                self.opt = tf.train.GradientDescentOptimizer(rate)
                self.grads_and_vars = self.opt.compute_gradients(self.loss)
                self.grads_and_vars = [[[g, v] for g, v in self.grads_and_vars
                                       if g is not None]]
                self.update = self.opt.apply_gradients(self.grads_and_vars)
