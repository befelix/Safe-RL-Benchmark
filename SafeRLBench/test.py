"""
SafeRLBench module testing.

These tests do not test specific behavior yet or ensure correctness in any way,
since the monitor may be suspect to large changes soon.
However a high coverage is able to detect some kind of problems in case parts
of the code are not used a lot, when the library is used correctly and may thus
bring weak parts to the attention of the programmer.
"""

from __future__ import division, print_function, absolute_import

# Monitor testing imports
from SafeRLBench import config
from SafeRLBench.monitor import EnvMonitor, AlgMonitor

# Benchmark testing imports
from SafeRLBench import Bench, BenchConfig
from SafeRLBench.bench import BenchRun
from SafeRLBench.algo import PolicyGradient
from SafeRLBench.envs import LinearCar

# General testing imports
from mock import Mock, MagicMock, patch
from unittest import TestCase

import logging

logger = logging.getLogger(__name__)


def _check_attr_impl(obj, attr_list):
    for attr in attr_list:
        assert(hasattr(obj, attr))


# TODO: Monitor testing needs improvement
class TestMonitor(TestCase):
    """Monitor module testing..."""

    def testMonitorInit(self):
        """Test Monitor initialization..."""
        attr_list = ['verbose', 'data']
        monitor = config.monitor
        _check_attr_impl(monitor, attr_list)

    def testEnvMonitorInit(self):
        """Test EnvMonitor initialization."""
        attr_list = ['rollout_cnt']
        envmonitor = EnvMonitor()
        _check_attr_impl(envmonitor, attr_list)

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
        _check_attr_impl(algmonitor, attr_list)

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
        self.assertRaises(RuntimeError, monitor.after_optimize, [alg_mock])

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


# TODO: Test the base.
class TestBase(object):
    pass


class TestBench(TestCase):
    """Bench tests."""

    def testBenchInit(self):
        """Test Bench initialization."""
        bench = Bench()

        self.assertIsInstance(bench.config, BenchConfig)
        self.assertIsInstance(bench.runs, list)

        bench = Bench(BenchConfig())

        self.assertIsInstance(bench.config, BenchConfig)
        self.assertIsInstance(bench.runs, list)

    @patch('SafeRLBench.bench.BenchRun')
    def testBenchBenchmark(self, bench_run_mock):
        """Test Bench benchmark invokation."""
        # setup mocks
        bench_run_obj_mock = Mock()
        bench_conf_mock = MagicMock(spec=BenchConfig)

        def create_run_obj_mock(a, b, c, d):
            return bench_run_obj_mock

        bench_run_mock.side_effect = create_run_obj_mock
        bench_conf_mock.__iter__.return_value = [(Mock(), Mock(), {}, {})]

        bench = Bench(bench_conf_mock)
        bench()

        bench_run_obj_mock.alg.optimize.assert_called_once_with()


class TestBenchConfig(TestCase):
    """BenchConfig tests."""

    # setup test configuration
    alg_config = [[
        (PolicyGradient, [{}]),
        (PolicyGradient, {})
    ], [
        (PolicyGradient, {})
    ]]

    env_config = [
        (LinearCar, {'horizon': 100}),
        (LinearCar, {'horizon': 200})
    ]

    alg_config_add = [
        (PolicyGradient, [{}, {}]),
    ]

    env_config_add = [
        (LinearCar, {'horizon': 100}),
        (LinearCar, {'horizon': 200})
    ]

    @staticmethod
    def _check_structure(lst):
        # loop through entire structure checking types.
        assert(isinstance(lst, list))
        for lst_elem in lst:
            assert(isinstance(lst_elem, list))
            for tup_elem in lst_elem:
                assert(isinstance(tup_elem, tuple))
                assert (tup_elem[0] is PolicyGradient
                        or tup_elem[0] is LinearCar)
                assert(isinstance(tup_elem[1], list))
                for dict_elem in tup_elem[1]:
                    assert(isinstance(dict_elem, dict))

    def testBenchConfigInit(self):
        """Test Bench Config initialization structure."""
        # apply test configuration
        config = BenchConfig(self.alg_config, self.env_config)

        # verify structure
        self._check_structure(config.algs)
        self._check_structure(config.envs)

    def testBenchConfigAddTests(self):
        """Test BenchConfig addTests."""
        # setup test configuration
        config = BenchConfig()

        # apply test configuration
        config.addTests(self.alg_config_add, self.env_config_add)

        # verify structure
        self._check_structure(config.algs)
        self._check_structure(config.envs)

    def testBenchConfigExceptions(self):
        """Test BenchConfig exceptions."""
        # setup bad test configurations
        alg_bad_tuple = [PolicyGradient, {}]
        env_bad_tuple = (LinearCar, {})
        bad_tuple = [alg_bad_tuple, env_bad_tuple]

        alg_bad_alg = [(Mock(), {})]
        env_bad_alg = [(LinearCar, {})]
        bad_alg = [alg_bad_alg, env_bad_alg]

        alg_bad_env = [(PolicyGradient, {})]
        env_bad_env = [(Mock, {})]
        bad_env = [alg_bad_env, env_bad_env]

        alg_bad_len = [(PolicyGradient, {})]
        env_bad_len = []
        bad_len = [alg_bad_len, env_bad_len]

        tests = [bad_tuple, bad_alg, bad_env, bad_len]

        # apply tests
        for test in tests:
            with self.subTest(test=test):
                self.assertRaises(ValueError, BenchConfig, *test)

    def testBenchConfigIterator(self):
        """Test BenchConfig Iterator."""
        config = BenchConfig(self.alg_config, self.env_config)

        for alg, env, alg_conf, env_conf in config:
            assert alg is PolicyGradient
            assert env is LinearCar
            self.assertIsInstance(alg_conf, dict)
            self.assertIsInstance(env_conf, dict)


class TestBenchRun(TestCase):
    """Test BenchRun class."""

    def testBenchRunInit(self):
        """Test Bench Run initialization."""
        args = [MagicMock() for i in range(4)]
        attr = ['alg', 'env', 'alg_conf', 'env_conf']

        run = BenchRun(*args)

        for a, m in zip(attr, args):
            assert getattr(run, a) is m

    def testBenchGetMonitor(self):
        """Test Monitor getters."""
        args = [MagicMock() for i in range(4)]
        run = BenchRun(*args)

        run.getAlgMonitor()
        run.getEnvMonitor()

        alg_mock = args[0]
        env_mock = args[1]

        alg_mock.monitor.__getitem__.assert_called_once_with(alg_mock)
        env_mock.monitor.__getitem__.assert_called_once_with(env_mock)
