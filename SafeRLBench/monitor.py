"""Monitoring implementations."""

import logging
import time

from SafeRLBench import config

logger = logging.getLogger(__name__)


class EnvMonitor(object):
    """
    Environment Monitor, providing tracking for environments.

    Methods
    -------
    _before_update
    _after_update
    _before_rollout
    _after_rollout
    _before_reset
    _after_reset
    """

    def __init__(self):
        """Initialize attributes."""
        self.monitor = EnvData()

    def _before_update(self):
        """
        Monitor environment before update.

        Parameters
        ----------
        env :
            Environment instance to be monitored.
        """
        pass

    def _after_update(self):
        """
        Monitor environment after update.

        Parameters
        ----------
        env :
            Environment instance to be monitored.
        """
        pass

    def _before_rollout(self):
        """
        Monitor environment before rollout.

        Parameters
        ----------
        env :
            Environment instance to be monitored.
        """
        pass

    def _after_rollout(self):
        """
        Monitor environment after rollout.

        Parameters
        ----------
        env :
            Environment instance to be monitored.
        """
        self.monitor.rollout_cnt += 1

    def _before_reset(self):
        """
        Monitor environment before reset.

        Parameters
        ----------
        env :
            Environment instance to be monitored.
        """
        pass

    def _after_reset(self):
        """
        Monitor environment after reset.

        Parameters
        ----------
        env :
            Environment instance to be monitored.
        """
        pass


class AlgoMonitor(object):
    """
    Algorithm Monitor, providing tracking capabilities for algorithms.

    Methods
    -------
    _before_optimize
    _after_optimize
    _before_step
    _after_step
    """

    def __init__(self):
        """Initialize attributes."""
        self.monitor = AlgoData()

    def _before_optimize(self):
        """Setup montitor for optimization run.

        Parameters
        ----------
        alg :
            the algorithm instance to be monitored
        """
        if config.monitor_verbosity > 0:
            logger.info('Starting optimization of %s...', str(self))

        # init monitor dict for algorithm
        self.monitor.t = time.time()

        # init optimization time control
        self.monitor.optimize_start = time.time()

    def _after_optimize(self):
        """Catch data after optimization run."""
        # retrieve time of optimization
        optimize_end = time.time()
        optimize_time = optimize_end - self.monitor.optimize_start

        if self.monitor.optimize_start == 0:
            logger.warning('Time measure for optimize corrupted')

        self.monitor.optimize_start = 0

        self.monitor.optimize_time = optimize_time

        logger.debug('Finished optimization after %d steps with grad %s.',
                     self.monitor.step_cnt, str(self.grad))

        # independently compute traces after optimization is finished
        if config.monitor_verbosity > 0:
            logger.info('Computing traces for %s run...', str(self))

        for parameter in self.monitor.parameters:

            self.policy.parameter = parameter

            # compute trace
            trace = self.environment._rollout(self.policy)
            self.monitor.traces.append(trace)

            # compute total reward
            reward = sum([t[2] for t in trace])
            self.monitor.rewards.append(reward)

    def _before_step(self):
        """Monitor algorithm before step.

        Parameters
        ----------
        alg :
            Algorithm instance to be monitored.
        """
        # count the number of rollouts for each step
        self.environment.monitor.rollout_cnt = 0

        if config.monitor_verbosity > 2:
            logger.info('Computing step %d for %s...', self.monitor.step_cnt,
                        str(self))

    def _after_step(self):
        """Monitor algorithm after step.

        Parameters
        ----------
        alg :
            Algorithm instance to be monitored.
        """
        emonitor = self.environment.monitor

        self.monitor.step_cnt += 1

        # store the number of rollouts since before step
        self.monitor.rollout_cnts.append(emonitor.rollout_cnt)

        # retrieve information
        parameter = self.policy.parameter

        # store information
        self.monitor.parameters.append(parameter)

        # log if wanted
        self._step_log()

    def _step_log(self):
        # print information if wanted
        monitor = self.monitor
        n = monitor.step_cnt
        log = 0

        # check verbosity level
        if config.monitor_verbosity > 0:
            if monitor.step_cnt % 1000 == 0:
                log = 1000

        if config.monitor_verbosity > 1:
            if monitor.step_cnt % 100 == 0:
                log = 100

        if config.monitor_verbosity > 2:
            log = 1

        if log:
            # generate time strings
            now = time.time()
            t = now - monitor.optimize_start
            t_s = "{:.2f}".format(t)
            avg_s = "{:.3f}".format(t / n)

            # retrieve current state
            par_s = str(self.policy.parameter)

            # generate log message
            msg = 'Status for ' + self.__class__.__name__ + ' on '
            msg += self.environment.__class__.__name__ + ':\n\n'
            msg += '\tRun: %d\tTime: %s\t Avg: %s\n' % (n, t_s, avg_s)
            msg += '\tParameter: \t%s\n' % (par_s)

            logger.info(msg)


class EnvData(object):
    """Class to store environment tracking data.

    Attributes
    ----------
    rollout_cnt : Int
        number of rollouts performed on environment.
    """

    def __init__(self):
        """Initialize attributes."""
        self.rollout_cnt = 0


class AlgoData(object):
    """Class used to store algorithm tracking data.

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
