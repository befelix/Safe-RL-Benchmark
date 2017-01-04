"""Tests for envs module."""
from __future__ import absolute_import

from numpy.random import randint
# import unittest
# from numpy.testing import *
import inspect
from functools import partial

import SafeRLBench.envs as envs
from SafeRLBench.tools import Policy


# TODO: Isolate unittests with mocks.
class TestEnvironments(object):
    """
    Test Class for Environment tests.

    Note that you really dont want to inherit from unittest.TestCase here,
    because it will break reasonable output with verbose testing.
    """

    classes = None

    @classmethod
    def setupClass(cls):
        """Generate list of classes."""
        cls.classes = []
        for name, c in inspect.getmembers(envs):
            if inspect.isclass(c):
                cls.classes.append(c)

    def test_environment_requirements(self):
        """Generate tests for environment implementations."""
        for c in self.classes:
            # Generate NotImplementedError Test for _update
            check_update = partial(self.check_env_update)
            check_update.description = "Check update implementation for "
            check_update.description += c.__name__
            yield check_update, c

            # Generate NotImplementedError Test for _reset
            check_reset = partial(self.check_env_reset)
            check_reset.description = "Check reset implementation for "
            check_reset.description += c.__name__
            yield check_reset, c

            check_rollout = partial(self.check_env_rollout)
            check_rollout.description = "Check rollout implementation for "
            check_rollout.description += c.__name__
            yield check_rollout, c

    def check_env_update(self, c):
        """Check if _update is implemented."""
        env = c()
        x = env.action_space.element()
        try:
            env._update(x)
        except NotImplementedError:
            assert False

    def check_env_reset(self, c):
        """Check if _reset is implemented."""
        env = c()
        try:
            env._reset()
        except NotImplementedError:
            assert False

    def check_env_rollout(self, c):
        """Check rollout correctness at random positions."""
        env = c()

        def par_policy(parameter):
            def policy(state):
                return env.state_space.element()[0]
            return policy

        policy = Policy(par_policy, (1,))
        policy.setParameter(None)
        trace = env._rollout(policy)

        tests = 500
        horizon = len(trace) - 1
        for n in range(tests):
            idx = randint(1, horizon)
            env.state = (trace[idx - 1])[1].copy()
            t = trace[idx]
            t_verify = env._update(t[0])
            assert(t_verify[0] == t[0])
            assert(all(t_verify[1] == t[1]))
            assert(t_verify[2] == t[2])

# TODO: Add better tests
