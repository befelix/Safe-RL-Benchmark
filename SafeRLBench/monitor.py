"""Monitoring implementations."""

import logging
import time

from weakref import WeakKeyDictionary

try:
    from collections import UserDict
except ImportError:
    # python 2.7 compatibility :(
    from UserDict import UserDict

logger = logging.getLogger(__name__)


class Monitor(UserDict):
    """
    This class is used to track algorithms and environments.

    Attributes
    ----------

    data : WeakKeyDictionary
        Dictionary used by UserDict. Keys need to be weakrefs to avoid cyclic
        references.

    Methods
    -------

    before_update
    after_update
    before_rollout
    after_rollout
    before_reset
    after_reset
    before_optimize
    after_optimize
    before_step
    after_step
    """

    def __init__(self, verbose=0):
        """
        Initialize Monitor.

        Parameters
        ----------
        verbosity :
            Verbosity level. Default 0.
        """
        self.verbose = verbose

        self.data = WeakKeyDictionary()

    def __getitem__(self, key):
        # this import can not be module wide because we import monitor before
        # base in the __init__.py file.
        from SafeRLBench import AlgorithmBase, EnvironmentBase

        if key in self.data:
            value = self.data[key]
        elif isinstance(key, AlgorithmBase):
            value = AlgMonitor()
            self.data[key] = value
        elif isinstance(key, EnvironmentBase):
            value = EnvMonitor()
            self.data[key] = value
        else:
            raise KeyError('Key has to be an algorithm or environment.')

        return value

    def before_update(self, env):
        """
        Monitor environment before update.

        Parameters
        ----------
        env :
            Environment instance to be monitored.
        """
        pass

    def after_update(self, env):
        """
        Monitor environment after update.

        Parameters
        ----------
        env :
            Environment instance to be monitored.
        """
        pass

    def before_rollout(self, env):
        """
        Monitor environment before rollout.

        Parameters
        ----------
        env :
            Environment instance to be monitored.
        """
        if env not in self.data:
            self[env] = EnvMonitor()

    def after_rollout(self, env):
        """
        Monitor environment after rollout.

        Parameters
        ----------
        env :
            Environment instance to be monitored.
        """
        self[env].rollout_cnt += 1

    def before_reset(self, env):
        """
        Monitor environment before reset.

        Parameters
        ----------
        env :
            Environment instance to be monitored.
        """
        pass

    def after_reset(self, env):
        """
        Monitor environment after reset.

        Parameters
        ----------
        env :
            Environment instance to be monitored.
        """
        pass

    def before_optimize(self, alg):
        """
        Setup montitor for optimization run.

        Parameters
        ----------
        alg :
            the algorithm instance to be monitored
        """
        if self.verbose > 0:
            logger.info('Starting optimization of %s...', str(alg))

        # init monitor dict for algorithm
        monitor = AlgMonitor()
        monitor.policy = alg.policy
        monitor.t = time.time()

        self[alg] = monitor

        if alg.environment not in self.data:
            self[alg.environment] = EnvMonitor()

        # init optimization time control
        monitor.optimize_start = time.time()

    def after_optimize(self, alg):
        """Catch data after optimization run."""
        try:
            monitor = self[alg]
        except KeyError:
            raise RuntimeError('before_optimize has not been called before '
                               + 'after_optimize.')
        # retrieve time of optimization
        optimize_end = time.time()
        optimize_time = optimize_end - monitor.optimize_start

        if monitor.optimize_start == 0:
            logger.warning('Time measure for optimize corrupted')

        monitor.optimize_start = 0

        monitor.optimize_time = optimize_time

        # independently compute traces after optimization is finished
        if self.verbose > 0:
            logger.info('Computing traces for %s run...', str(alg))

        policy = monitor.policy

        for parameter in monitor.parameters:

            policy.parameter = parameter

            # compute trace
            trace = alg.environment._rollout(policy)
            monitor.traces.append(trace)

            # compute total reward
            reward = sum([t[2] for t in trace])
            monitor.rewards.append(reward)

    def before_step(self, alg):
        """
        Monitor algorithm before step.

        Parameters
        ----------
        alg :
            Algorithm instance to be monitored.
        """
        # count the number of rollouts for each step
        self[alg.environment].rollout_cnt = 0

        if self.verbose > 2:
            logger.info('Computing step %d for %s...', self[alg].step_cnt,
                        str(alg))

    def after_step(self, alg):
        """
        Monitor algorithm after step.

        Parameters
        ----------
        alg :
            Algorithm instance to be monitored.
        """
        monitor = self[alg]
        emonitor = self[alg.environment]

        monitor.step_cnt += 1

        # store the number of rollouts since before step
        monitor.rollout_cnts.append(emonitor.rollout_cnt)

        # retrieve information
        policy = monitor.policy
        parameter = policy.parameter

        # store information
        monitor.parameters.append(parameter)

        # log if wanted
        self._step_log(alg)

    def _step_log(self, alg):
        # print information if wanted
        monitor = self[alg]
        n = monitor.step_cnt
        log = 0

        # check verbosity level
        if self.verbose > 0:
            if monitor.step_cnt % 1000 == 0:
                log = 1000

        if self.verbose > 1:
            if monitor.step_cnt % 100 == 0:
                log = 100

        if self.verbose > 2:
            log = 1

        if log:
            # generate time strings
            now = time.time()
            t = now - monitor.optimize_start
            t_s = "{:.2f}".format(t)
            avg_s = "{:.3f}".format(t / n)

            # retrieve current state
            par_s = str(alg.policy.parameter)
            grad_s = str(alg.grad)

            # generate log message
            msg = 'Status for ' + alg.__class__.__name__ + ' on '
            msg += alg.environment.__class__.__name__ + ':\n\n'
            msg += '\tRun: %d\tTime: %s\t Avg: %s\n' % (n, t_s, avg_s)
            msg += '\tParameter: \t%s\n\tGradient: \t%s\n' % (par_s, grad_s)

            logger.info(msg)


class EnvMonitor(object):
    """
    Class to store environment tracking data.

    Attributes
    ----------
    rollout_cnt : Int
        number of rollouts performed on environment.
    """

    def __init__(self):
        """Initialize attributes."""
        self.rollout_cnt = 0


class AlgMonitor(object):
    """
    Class used to store algorithm tracking data.

    Attributes
    ----------
    optimize_start : Float
        Start time of the optimization.
    optimize_time : Float
        Start time of intermediate runs.
    step_cnt : Int
        Number of steps performed since initialization.
    rollout_cnts : List
        Number of rollouts during one step.
    parameters : List
        List of parameters found during optimization.
    traces : List
        List of traces for parameters.
    rewards : List
        List of rewards for parameters.
    """

    def __init__(self):
        """Initialize attributes."""
        self.optimize_start = 0
        self.optimize_time = 0

        self.step_cnt = 0
        self.rollout_cnts = []

        self.parameters = []
        self.traces = []
        self.rewards = []
