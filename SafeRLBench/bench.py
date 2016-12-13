from SafeRLBench import EnvironmentBase, AlgorithmBase

import logging

logger = logging.getLogger(__name__)

__all__ = ['Bench']


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
            algos = dict(algos)
            if check_algos(algos):
                self.algos = algos
            else:
                raise ValueError("Argument algos contains invalid element.")

        if envs is None:
            self.envs = []
        else:
            envs = dict(envs)
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

        self.tests = []

    def __call__(self):
        self.benchmark()

    def benchmark(self):

        # Initialize all tests
        self._instantiateObjects()

        # Run tests
        # TODO: Add parallelism support
        self._runTests()

    def _runTests(self):

        for test in self.tests:
            test.optimize()

    def _instantiateObjects(self):
        # loop over all possible combinations
        for alg in self.algos:
            for env in self.envs:
                for (alg_conf, env_conf) in self.configs[alg, env]:
                    env_obj = env(**env_conf)
                    alg_obj = alg(env_obj, **alg_conf)

                    self.tests.append(alg_obj)

        logger.info("Initialized %d tests.", len(self.tests))

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
