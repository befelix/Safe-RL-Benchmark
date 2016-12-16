"""Benchmarking facilities."""

from SafeRLBench import EnvironmentBase, AlgorithmBase

from itertools import product

import logging
import pprint

logger = logging.getLogger(__name__)

__all__ = ('Bench', 'BenchConfig')


def check_algos(algos):
    for alg in algos:
        if not issubclass(alg, AlgorithmBase):
            return False
    return True


def check_envs(envs):
    for env in envs:
        if not issubclass(env, EnvironmentBase):
            return False
    return True


class Bench(object):
    """
    Benchmarking class to benchmark algorithms on various environments.

    Attributes
    ----------
    algos : None
        List of alorithms to be tested.
    envs : None
        List of environment to be tested.
    config :
        Dictionary containing lists of configurations for every (alg, env)
        tuple.
    measures :
        Not clear yet
    """

    def __init__(self, algos=None, envs=None, configs=None, measures=None):
        """
        Initialize Bench instance.

        Parameters
        ----------
        algos :
            List of algorithms for benchmarking. Default is None.
        envs :
            List of environment for benchmarking. Default is None.
        configs : BenchConfig instance
            BenchConfig information supplying configurations. Default is None.
        """
        if algos is None:
            self.algos = []
        else:
            if check_algos(algos):
                self.algos = algos
            else:
                raise ValueError("Argument algos contains invalid element.")

        if envs is None:
            self.envs = []
        else:
            if check_envs(envs):
                self.envs = envs
            else:
                raise ValueError("Argument envs contains invalid element.")

        if configs is None:
            self.configs = {}
        else:
            self.configs = configs

        if measures is None:
            self.measures = []
        else:
            self.measures = measures

        self.runs = []

    def __call__(self):
        """Initialize and run benchmark as configured."""
        self.benchmark()

    def benchmark(self):
        """Initialize and run benchmark as configured."""
        logger.debug('Starting benchmarking.')

        for alg, env, alg_conf, env_conf in self.configs:
            env_obj = env(**env_conf)
            alg_obj = alg(env_obj, **alg_conf)

            run = BenchRun(alg_obj, env_obj, alg_conf, env_conf)
            self.runs.append(run)

            logger.debug('DISPATCH RUN:\n\n%s\n', str(run))
            run.alg.optimize()

    def addAlgorithm(self, alg):
        if issubclass(alg, AlgorithmBase):
            self.algos.append(alg)
        else:
            logger.warning("No algorithm added. Argument invalid.")

    def addEnvironment(self, env):
        if issubclass(env, EnvironmentBase):
            self.envs.append(env)
        else:
            logger.warning("No environment added. Argument invalid.")

    def addMeasure(self):
        pass

    def plotMeasure(self):
        pass


class BenchConfig(object):
    """
    Benchmark configuration class.

    This class is supposed to provide a convenient interface to setup
    configurations for benchmarking runs.
    When we are defining the configurations for benchmarking we face the major
    inconvenience that an algorithm's configuration may depend on the
    environment configurations. In the case where we want to benchmark multiple
    algorithms on multiple environments this requires many redunant definitions
    if done manually.

    We thus assume there are two major cases:
    * One environment configuration for multiple algorithm configurations.
    * One algorithm configuration for multiple environment configurations.

    Attributes
    ----------
    algs :
        List of list of tuples where the first element is an algorithm and
        the second a list of configurations. Any inner list may also be a
        single element instead of a list.

    envs :
        List of tuples where the first element is an environment and the
        second a configuration.

    Examples
    --------

    """

    def __init__(self, algs, envs):
        """
        Initialize BenchConfig instance.

        This initializer may be used if you want to test multiple algorithm
        configurations on a list of environments.

        Parameters
        ----------
        algs :
            List of list of tuples where the first element is an algorithm and
            the second a list of configurations. Any inner list may also be a
            single element instead of a list.

        envs :
            List of tuples where the first element is an environment and the
            second a configuration.
        """
        if (len(algs) != len(envs)):
            raise ValueError('Configuration lists dont have same length')

        for i, obj in enumerate(algs):
            algs[i] = self._listify(obj)

        for i, obj in enumerate(envs):
            envs[i] = self._listify(obj)

        self.algs = algs
        self.envs = envs

    def addEnvTests(self, algs, envs):
        """
        Adds one environment configuration and algorithm configurations to be
        run on it.

        Parameters
        ----------
        algs :
            List of tuples where the first element is an algorithm and the
            second a list of configurations. May also be single elements.
        env :
            tuple of environment and list of configurations or configuration
            dictionary.
        """
        algs = self._listify(algs)
        envs = self._listify(envs)

        self.algs.append(algs)
        self.envs.append(envs)

    def _listify(self, obj):
        if not isinstance(obj, list):
            obj = [obj]
        for i, tup in enumerate(obj):
            if not isinstance(tup[1], list):
                obj[i] = (tup[0], [tup[1]])
        return obj

    def __iter__(self):
        for algs, envs in zip(self.algs, self.envs):
            for (alg, alg_confs), (env, env_confs) in product(algs, envs):
                for alg_conf, env_conf in product(alg_confs, env_confs):
                    yield alg, env, alg_conf, env_conf


class BenchRun(object):
    """
    Wrapper containing instances and configuration for a run.

    Attributes
    ----------
    alg :
        Algorithm instance

    env :
        Environment instance

    alg_conf : Dictionary
        Algorithm configuration

    env_conf : Dictionary
        Environment configuration
    """

    def __init__(self, alg, env, alg_conf, env_conf):
        """
        Initialize BenchRun instance.

        Parameters
        ----------
        alg :
            Algorithm instance
        env :
            Environment instance
        alg_conf : Dictionary
            Algorithm configuration
        env_conf : Dictionary
            Environment configuration
        """
        self.alg = alg
        self.env = env

        self.alg_conf = alg_conf
        self.env_conf = env_conf

    def getAlgMonitor(self):
        """Retrieve AlgMonitor for algorithm."""
        return self.alg.monitor[self.alg]

    def getEnvMonitor(self):
        """Retrieve EnvMonitor for environment."""
        return self.env.monitor[self.env]

    def __repr__(self):
        out = []
        out += ['Algorithm: ', [self.alg.__class__.__name__, self.alg_conf]]
        out += ['Environment: ', [self.env.__class__.__name__, self.env_conf]]
        trans_dict = {ord(c): ord(' ') for c in ',\'\[\]'}
        return pprint.pformat(out, indent=2).translate(trans_dict)
