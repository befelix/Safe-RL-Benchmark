"""
SafeRLBench module testing.

These tests do not test specific behavior yet or ensure correctness in any way,
since the monitor may be suspect to large changes soon.
However a high coverage is able to detect some kind of problems in case parts
of the code are not used a lot, when the library is used correctly and may thus
bring weak parts to the attention of the programmer.
"""

from __future__ import division, print_function, absolute_import

# import unittest
# from numpy.testing import *
from SafeRLBench import config
from SafeRLBench.monitor import EnvMonitor, AlgMonitor
from mock import Mock


def check_attr_impl(obj, attr_list):
    for attr in attr_list:
        assert(hasattr(obj, attr))


class TestMonitor(object):
    """Monitor module testing..."""

    def testMonitorInit(self):
        """Test Monitor initialization..."""
        attr_list = ['verbose', 'data']
        monitor = config.monitor
        check_attr_impl(monitor, attr_list)

    def testEnvMonitorInit(self):
        """Test EnvMonitor initialization."""
        attr_list = ['rollout_cnt']
        envmonitor = EnvMonitor()
        check_attr_impl(envmonitor, attr_list)

    def testAlgMonitorInit(self):
        """Test AlgMonitor initialization."""
        attr_list = [
            'optimize_start',
            'optimize_time',
            'step_cnt',
            'rollout_cnts',
            'parameters',
            'traces',
            'rewards',
        ]
        algmonitor = AlgMonitor()
        check_attr_impl(algmonitor, attr_list)

    def testMonitorMethods(self):
        """Test Monitor method calls."""
        env_mock = Mock()
        alg_mock = Mock()

        config.monitorSetVerbosity(3)
        monitor = config.monitor

        # environment monitor checks
        monitor.before_update(env_mock)
        monitor.after_update(env_mock)
        monitor.before_rollout(env_mock)
        monitor.after_rollout(env_mock)
        monitor.before_reset(env_mock)
        monitor.after_reset(env_mock)

        # algorithm monitor checks
        try:
            monitor.after_optimize(alg_mock)
            assert(False)
        except:
            pass

        monitor.before_optimize(alg_mock)
        monitor.before_step(alg_mock)
        monitor.after_step(alg_mock)

        # test specific conditionals
        def rollout_mock(*args):
            return [(None, None, 1), (None, None, 2)]

        alg_mock.environment._rollout = rollout_mock

        monitor[alg_mock] = Mock()
        parameter_mock = Mock()
        monitor[alg_mock].parameters = [parameter_mock]

        monitor[alg_mock].step_cnt = 999
        monitor[alg_mock].optimize_start = 0

        monitor.after_step(alg_mock)
        monitor.after_optimize(alg_mock)


class TestBase(object):
    pass


class TestBench(object):
    pass


class TestConfig(object):
    pass

# TODO: Add tests
