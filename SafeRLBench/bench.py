"""Benchmarking facilities."""
from SafeRLBench import EnvironmentBase, AlgorithmBase

try:
    from collections import UserDict
except:
    from UserDict import UserDict

import logging
import pprint

logger = logging.getLogger(__name__)

__all__ = ['Bench', 'BenchConfig']


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
        self.benchmark()

    def benchmark(self):
        """Initialize and run benchmark as configured."""
        # Initialize all runs
        self._instantiateObjects()

        # Run tests
        # TODO: Add parallelism support
        self._runTests()

    def _runTests(self):
        for run in self.runs:
            logger.debug('DISPATCH RUN:\n\n%s\n',
                         str(run))
            run.alg.optimize()

    def _instantiateObjects(self):
        # loop over all possible combinations
        for alg in self.algos:
            for env in self.envs:
                if not self.configs[alg, env]:
                    logger.warning('No configuration for (%s, %s)',
                                   alg.__class__.__name__,
                                   env.__class__.__name__)

                for (alg_conf, env_conf) in self.configs[alg, env]:
                    env_obj = env(**env_conf)
                    alg_obj = alg(env_obj, **alg_conf)

                    run = BenchRun(alg_obj, env_obj, (alg_conf, env_conf))

                    self.runs.append(run)

        logger.info("Initialized %d runs.", len(self.runs))

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


class BenchConfig(UserDict):
    def __init__(self):
        self.data = {}

    def addAlgConfig(self, alg, env, alg_confs, env_conf={}):
        """
        Add an algorithm configuration for an environment setup.

        Parameters
        ----------
        alg :
            Algorithm class
        env :
            Environment class
        alg_configs :
            List of configuration for algorithm
        env_config :
            Configuration for environment
        """
        if (alg, env) not in self:
            self[alg, env] = []

        for alg_conf in alg_confs:
            self[alg, env].append((alg_conf, env_conf))


class BenchRun(object):
    """
    Wrapper containing instances and configuration for a run.

    Attributes
    ----------
    alg :
        Algorithm instance
    env :
        Environment instance
    config : (Dict, Dict)
        Tuple of configurations for algorithm and environment
    """
    def __init__(self, alg, env, config):
        self.alg = alg
        self.env = env

        self.config = config

    def getAlgMonitor(self):
        """Retrieve AlgMonitor for algorithm."""
        return self.alg.monitor[self.alg]

    def getEnvMonitor(self):
        """Retrieve EnvMonitor for environment."""
        return self.env.monitor[self.env]

    def __repr__(self):
        out = []
        out += ['Algorithm: ', [self.alg.__class__.__name__, self.config[0]]]
        out += ['Environment: ', [self.env.__class__.__name__, self.config[1]]]
        trans_dict = {ord(c): ord(' ') for c in ',\'\[\]'}
        return pprint.pformat(out, indent=2).translate(trans_dict)
