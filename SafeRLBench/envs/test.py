"""Tests for envs module.

Need rework.
"""
from __future__ import absolute_import

# import unittest
# from numpy.testing import *
import inspect
from functools import partial

import SafeRLBench.envs as envs

import numpy as np

import gym
gym.undo_logger_setup()

from mock import Mock


class TestEnvironments(object):
    """
    Test Class for Environment tests.

    Note that you really dont want to inherit from unittest.TestCase here,
    because it will break reasonable output with verbose testing.
    """

    exclude = []
    not_deterministic = ['Quadrocopter']

    args = {
        'GymWrap': envs.gym_wrap._get_test_args(),
        'MDP': envs.mdp._get_test_args()
    }

    @classmethod
    def setUpClass(cls):
        """Generate list of classes."""
        cls.classes = []
        for name, c in inspect.getmembers(envs):
            if inspect.isclass(c):
                cls.classes.append(c)

    def test_environment_requirements(self):
        """Generate tests for environment implementations."""
        for c in self.classes:
            if c.__name__ in self.exclude:
                pass
            else:
                # Generate NotImplementedError Test for _update
                check_update = partial(self.check_env_update)
                check_update.description = ('Test: ' + c.__name__.upper()
                                            + ': update implementation.')
                yield check_update, c

                # Generate NotImplementedError Test for _reset
                check_reset = partial(self.check_env_reset)
                check_reset.description = ('Test: ' + c.__name__.upper()
                                           + ': reset implementation.')
                yield check_reset, c

                check_rollout = partial(self.check_env_rollout)
                check_rollout.description = ('Test: ' + c.__name__.upper()
                                             + ': rollout implementation.')
                yield check_rollout, c

    def check_env_update(self, c):
        """Check if _update is implemented."""
        args = self.args.get(c.__name__, [])
        env = c(*args)
        x = env.action_space.sample()
        try:
            env._update(x)
        except NotImplementedError:
            assert False

    def check_env_reset(self, c):
        """Check if _reset is implemented."""
        args = self.args.get(c.__name__, [])
        env = c(*args)
        try:
            env._reset()
        except NotImplementedError:
            assert False

    def check_env_rollout(self, c):
        """Check rollout correctness at random positions."""
        if c.__name__ in self.not_deterministic:
            return

        args = self.args.get(c.__name__, [])
        env = c(*args)

        def policy(state):
            return env.action_space.sample()

        policy_mock = Mock(side_effect=policy)
        trace = env._rollout(policy_mock)

        # if the environment depends on a seed, reset it.
        if hasattr(env, 'seed'):
            env.seed = env.seed

        horizon = len(trace) - 1

        # in case we use a random state we need to do an initial iteration,
        # to obtain the same random state after reseeding.
        env._update(policy(None))

        for idx in range(1, horizon):
            env.state = (trace[idx - 1])[1].copy()
            t = trace[idx]
            t_verify = env._update(t[0])

            if isinstance(t[0], np.ndarray):
                assert(all(t_verify[0] == t[0]))
            else:
                assert(t_verify[0] == t[0])
            if isinstance(t[1], np.ndarray):
                assert(all(t_verify[1] == t[1]))
            else:
                assert(t_verify[1] == t[1])
            assert(t_verify[2] == t[2])
