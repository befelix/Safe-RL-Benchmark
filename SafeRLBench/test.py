"""
SafeRLBench module testing.

These tests do not test specific behavior yet or ensure correctness in any way,
since the monitor may be suspect to large changes soon.
However a high coverage is able to detect some kind of problems in case parts
of the code are not used a lot, when the library is used correctly and may thus
bring weak parts to the attention of the programmer.
"""

from __future__ import division, print_function, absolute_import

from SafeRLBench import config

# Benchmark testing imports
from SafeRLBench import Bench, BenchConfig
from SafeRLBench.measure import BestPerformance
from SafeRLBench.bench import BenchRun
from SafeRLBench.algo import PolicyGradient
from SafeRLBench.envs import LinearCar
from SafeRLBench.policy import LinearPolicy

# General testing imports
from mock import Mock, MagicMock, patch
from unittest2 import TestCase

import logging

logger = logging.getLogger(__name__)


def _check_attr_impl(obj, attr_list):
    for attr in attr_list:
        assert(hasattr(obj, attr))


class TestIntegration(TestCase):
    """Test integration with PolicyGradient and LinearCar."""

    # TODO: So far there is just a bm run to check if things work together.
    def test_integration(self):
        """Integration of PG and LC."""
        # setup config:
        config.logger_set_level(logging.DEBUG)
        config.monitor_set_verbosity(3)

        policy = LinearPolicy(2, 1, biased=True)
        algs = [(PolicyGradient, {'policy': policy,
                                  'max_it': 10,
                                  'estimator': 'central_fd'})]
        env = [[(LinearCar, {'horizon': 100})]]

        test_config = BenchConfig(algs, env)

        benchmark = Bench(test_config, [BestPerformance()])
        benchmark.benchmark()

        assert(benchmark.measures[0].result is not None)

    def test_parallel_integration(self):
        """Parallel integration of PG and LC."""
        # setup config:
        config.logger_set_level(logging.DEBUG)
        config.monitor_set_verbosity(3)
        config.jobs_set(2)

        policy = LinearPolicy(2, 1)
        algs = [(PolicyGradient, [{'policy': policy,
                                   'max_it': 10,
                                   'estimator': 'central_fd'},
                                  {'policy': policy,
                                   'max_it': 20,
                                   'estimator': 'central_fd'}])]
        env = [[(LinearCar, {'horizon': 100})]]

        test_config = BenchConfig(algs, env)

        benchmark = Bench(test_config, [BestPerformance()])
        benchmark.benchmark()

        assert(benchmark.measures[0].result is not None)
        assert(len(benchmark.measures[0].result) == 2)


# TODO: Monitor testing needs improvement
class TestMonitor(TestCase):
    """Monitor module testing..."""

    pass
    # TODO: Test Monitor


# TODO: Test the base.
class TestBase(object):
    """Test Base classes in base.py."""

    pass


class TestBench(TestCase):
    """Bench tests."""

    def test_bench_init(self):
        """Test Bench initialization."""
        bench = Bench()

        self.assertIsInstance(bench.config, BenchConfig)
        self.assertIsInstance(bench.runs, list)

        bench = Bench(BenchConfig())

        self.assertIsInstance(bench.config, BenchConfig)
        self.assertIsInstance(bench.runs, list)

    @patch('SafeRLBench.bench.BenchRun')
    def test_bench_benchmark(self, bench_run_mock):
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

    def test_benchconfig_init(self):
        """Test Bench Config initialization structure."""
        # apply test configuration
        config = BenchConfig(self.alg_config, self.env_config)

        # verify structure
        self._check_structure(config.algs)
        self._check_structure(config.envs)

    def test_benchconfig_add_tests(self):
        """Test BenchConfig add_tests."""
        # setup test configuration
        config = BenchConfig()

        # apply test configuration
        config.add_tests(self.alg_config_add, self.env_config_add)

        # verify structure
        self._check_structure(config.algs)
        self._check_structure(config.envs)

    def test_benchconfig_exceptions(self):
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

    def test_benchconfig_iterator(self):
        """Test BenchConfig Iterator."""
        conf = BenchConfig(self.alg_config, self.env_config)

        for alg, env, alg_conf, env_conf in conf:
            assert alg is PolicyGradient
            assert env is LinearCar
            self.assertIsInstance(alg_conf, dict)
            self.assertIsInstance(env_conf, dict)


class TestBenchRun(TestCase):
    """Test BenchRun class."""

    def test_benchrun_init(self):
        """Test Bench Run initialization."""
        args = [MagicMock() for i in range(4)]
        attr = ['alg', 'env', 'alg_conf', 'env_conf']

        run = BenchRun(*args)

        for a, m in zip(attr, args):
            assert getattr(run, a) is m

    def test_benchrun_get_monitor(self):
        """Test Monitor getters."""
        # TODO: Rewrite
        pass
