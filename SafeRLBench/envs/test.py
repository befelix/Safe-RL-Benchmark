"""Tests for envs module."""
from __future__ import absolute_import

# import unittest
# from numpy.testing import *
import inspect
from functools import partial

import SafeRLBench.envs as envs


def test_environment_requirements():
    """Generate tests for environment implementations."""
    for name, c in inspect.getmembers(envs):
        if inspect.isclass(c):
            # Generate NotImplementedError Test for _update
            check_update = partial(check_env_update)
            check_update.description = "Check update implementation for "
            check_update.description += c.__name__
            yield check_update, c

            # Generate NotImplementedError Test for _reset
            check_reset = partial(check_env_reset)
            check_reset.description = "Check reset implementation for "
            check_reset.description += c.__name__
            yield check_reset, c


def check_env_update(c):
    """Check if _update is implemented."""
    env = c()
    x = env.action_space.element()
    try:
        env._update(x)
    except NotImplementedError:
        assert False


def check_env_reset(c):
    """Check if _reset is implemented."""
    env = c()
    try:
        env._reset()
    except NotImplementedError:
        assert False


# TODO: Add more tests
