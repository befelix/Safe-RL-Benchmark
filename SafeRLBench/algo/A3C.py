"""Asynchronous Actor-Critic Agents.

Implementations refer to Denny Britz implementations at
https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient/a3c
"""

import copy

import numpy as np

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

    def __init__(self, env, policy, p_net, v_net, name):
        self.name = name
        self.env = copy.copy(env)
        self.global_policy = policy
        self.global_p_net = p_net
        self.global_v_net = v_net

        # generate local networks
        with tf.variable_scope(name):
            self.local_policy = policy.copy()
            self.local_p_net = _PolicyNet(self.local_policy,
                                          self.global_p_net.rate)
            self.local_v_net = _ValueNet(self.local_policy,
                                         self.global_v_net.rate)

        # create copy op
        trainable_variables = tf.GraphKeys.TRAINABLE_VARIABLES
        self.copy_params_op = self.make_copy_params_op(
            tf.contrib.slim.get_variables(scope="global",
                                          collection=trainable_variables),
            tf.contrib.slim.get_variables(scope=self.name,
                                          collection=trainable_variables))

        # create train ops
        self.p_net_train = self.make_train_op(self.local_p_net,
                                              self.global_p_net)
        self.v_net_train = self.make_train_op(self.local_v_net,
                                              self.global_v_net)

        self.state = self.env.state

    def run(self, sess, t_max):
        with sess.as_default():
            # maybe use eval, then this would not be required.
            self.local_policy.sess = sess

            sess.run(self.copy_params_op)

            # perform a rollout
            trace = self.env.rollout(self.policy)

            # compute advantage

    @staticmethod
    def make_copy_params_op(v1_list, v2_list):
        """Create operation to copy parameters.

        Creates an operation that copies parameters from variable in v1_list to
        variables in v2_list.
        The ordering of the variables in the lists must be identical.
        """
        v1_list = list(sorted(v1_list, key=lambda v: v.name))
        v2_list = list(sorted(v2_list, key=lambda v: v.name))

        update_ops = []
        for v1, v2 in zip(v1_list, v2_list):
            op = v2.assign(v1)
            update_ops.append(op)

        return update_ops

    @staticmethod
    def make_train_op(loc, glob):
        """Create operation that applies local gradients to global network."""
        loc_grads, _ = zip(*loc.grads_and_vars)
        loc_grads, _ = tf.clip_by_global_norm(loc_grads, 5.0)
        _, glob_vars = zip(*glob.grads_and_vars)
        loc_grads_glob_vars = list(zip(loc_grads, glob_vars))
        get_global_step = tf.contrib.framework.get_global_step()

        return glob.optimizer.apply_gradients(loc_grads_glob_vars,
                                              global_step=get_global_step)


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
