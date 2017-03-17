"""Asynchronous Actor-Critic Agents.

Implementations refer to Denny Britz implementations at
https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient/a3c
"""

import copy
import threading

import numpy as np

from SafeRLBench import AlgorithmBase

import SafeRLBench.error as error
from SafeRLBench.error import NotSupportedException

try:
    import tensorflow as tf
    from tensorflow.contrib.distributions import Normal
except:
    tf = None

import logging

logger = logging.getLogger(__name__)


class A3C(AlgorithmBase):
    """Implementation of the Asynchronous Actor-Critic Agents Algorithm.

    Attributes
    ----------
    environment :
        Environment we want to optimize the policy for.
    policy :
        The policy we want to optimize.
    max_it : integer
        The maximal number of iterations before we abort.
    num_workers : integer
        Number of workers that should be used asynchronous.
    rate : float
        Gradient Descent rate.
    discount : float
        Discount factor for adjusted reward.
    done : boolean
        Indicates whether the run is done.
    workers : list of _Worker instances
    threads : list of Thread instances.
    """

    def __init__(self, environment, policy, max_it=1000, num_workers=2,
                 rate=0.1, discount=0.1):
        """Initialize A3C."""
        if tf is None:
            raise NotSupportedException(error.NO_TF_SUPPORT)

        if policy.is_set_up:
            raise(ValueError('Policy should not be set up.'))

        super(A3C, self).__init__(environment, policy, max_it)

        self.num_workers = num_workers
        self.rate = rate
        self.discount = discount

        self.done = False

        self.policy = policy

        # init networks
        with tf.variable_scope('global'):
            self.p_net = _PolicyNet(self.policy, rate)
            self.v_net = _ValueNet(self.policy, rate)

        self.workers = []
        self.threads = []

        self.global_counter = 0

        self.sess = None

    def _initialize(self):
        self.global_counter = 0
        self.workers = []
        self.threads = []

        self.done = False

        for i in range(self.num_workers):
            worker = _Worker(self.environment,
                             self.policy,
                             self.p_net,
                             self.v_net,
                             self.discount,
                             'worker_' + str(i))
            self.workers.append(worker)

        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)

        # Write a graph file
        # TODO: enhance summary
        graph = self.sess.graph
        writer = tf.summary.FileWriter('logs/', graph=graph)
        writer.flush()

    def _step(self):
        self.global_counter += 1
        self.grad = [g for g, v in self.p_net.grads_and_vars]
        # TODO: Properly implement this.
        if self.global_counter % 100 == 0:
            logger.debug("Hey we are at step %d", self.global_counter)

    def _is_finished(self):
        if self.global_counter >= self.max_it:
            self.done = True
        return self.done

    def _optimize(self):
        self.initialize()
        for worker in self.workers:
            t = threading.Thread(target=self._start(worker, self.sess))
            self.threads.append(t)
            t.start()

        for t in self.threads:
            t.join()

    def _start(self, worker, sess):
        while not self.is_finished():
            p_net_loss, v_net_loss = worker.run(sess)
            self.step()

    # placeholder for now
    def _after_optimize(self):
        with self.sess.as_default():
            super(A3C, self)._after_optimize()


class _Worker(object):
    """Worker thread."""

    def __init__(self, env, policy, p_net, v_net, discount, name):
        self.name = name
        self.env = copy.copy(env)
        self.global_policy = policy
        self.global_p_net = p_net
        self.global_v_net = v_net

        self.discount = discount

        # generate local networks
        self.local_policy = policy.copy(name, do_setup=False)
        with tf.variable_scope(name):
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

    def run(self, sess):
        with sess.as_default():
            sess.run(self.copy_params_op)

            # perform a rollout
            trace = self.env.rollout(self.local_policy)

            advantages = []
            values = []
            states = []
            actions = []

            value = 0.

            for (action, state, reward) in trace:
                value = reward + self.discount * value

                # evaluate value net on state
                value_pred = sess.run(self.local_v_net.V_est,
                                      {self.local_v_net.X: [state]})
                advantage = reward - value_pred.flatten()

                advantages.append(advantage)
                values.append(value)
                states.append(state)
                actions.append(action)

            # compute local gradients and train global network
            feed_dict = {
                self.local_p_net.X: np.array(states),
                self.local_p_net.target: advantages,
                self.local_p_net.a: actions,
                self.local_v_net.X: np.array(states),
                self.local_v_net.V: values
            }

            p_net_loss, v_net_loss, _, _ = sess.run([
                self.local_p_net.loss,
                self.local_v_net.loss,
                self.p_net_train,
                self.v_net_train
            ], feed_dict)

        return p_net_loss, v_net_loss

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

        return glob.opt.apply_gradients(loc_grads_glob_vars,
                                        global_step=get_global_step)


class _ValueNet(object):
    """Wrapper for the Value function."""

    def __init__(self, policy, rate, train=True):
        self.rate = rate

        with tf.variable_scope('value_estimator'):
            self.X = tf.placeholder(policy.dtype,
                                    shape=policy.X.shape,
                                    name='X')
            self.V = tf.placeholder(policy.dtype,
                                    shape=[None],
                                    name='V')

            self.W = policy.init_weights((policy.layers[0], 1))

            self.V_est = tf.matmul(self.X, self.W)

            self.losses = tf.squared_difference(self.V_est, self.V)
            self.loss = tf.reduce_sum(self.losses, name='loss')

            if train:
                self.opt = tf.train.RMSPropOptimizer(rate, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.opt.compute_gradients(self.loss)
                self.grads_and_vars = [(g, v) for g, v in self.grads_and_vars
                                       if g is not None]
                self.update = self.opt.apply_gradients(self.grads_and_vars)


class _PolicyNet(object):
    """Wrapper for the Policy function."""

    def __init__(self, policy, rate, train=True):
        self.rate = rate
        self.policy = policy

        with tf.variable_scope('policy_estimator'):
            self.policy.setup()

            self.X = policy.X
            self.a = policy.a
            self.target = tf.placeholder(dtype='float', shape=[None, 1],
                                         name='target')

            self.a_pred = policy.a_pred
            self.var = policy.var

            dist = Normal(self.a_pred, self.var)
            self.log_probs = dist.log_pdf(self.a)
            """
            diff = tf.subtract(self.a, self.a_pred)
            fac = tf.div(tf.constant(1 / np.sqrt(2 * np.pi), 'float'),
                         self.var)
            self.probs = tf.multiply(tf.exp(tf.square(tf.div(diff, self.var))),
                                     fac)
            self.log_probs = tf.log(self.probs)
            """

            self.losses = - (self.log_probs * self.target)
            self.loss = tf.reduce_sum(self.losses, name='loss')

            if train:
                self.opt = tf.train.RMSPropOptimizer(rate, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.opt.compute_gradients(self.loss)
                self.grads_and_vars = [(g, v) for g, v in self.grads_and_vars
                                       if g is not None]
                self.update = self.opt.apply_gradients(self.grads_and_vars)
