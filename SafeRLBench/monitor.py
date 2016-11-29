import logging
import time

logger = logging.getLogger(__name__)


class Monitor(object):
    """This class is used to track algorithms and environments."""

    def __init__(self, verbose=0):
        self.verbose = verbose

        self.monitors = {}

    def before_update(self, env):
        pass

    def after_update(self, env):
        pass

    def before_rollout(self, env):
        if env not in self.monitors:
            self.monitors[env] = EnvMonitor()

    def after_rollout(self, env):
        self.monitors[env].rollout_cnt += 1

    def before_reset(self, env):
        pass

    def after_reset(self, env):
        pass

    def before_optimize(self, alg, policy):
        """Setup montitor for optimization run."""
        if self.verbose > 0:
            logger.info('Starting optimization of %s...', str(alg))

        # init monitor dict for algorithm
        monitor = AlgMonitor()
        monitor.policy = policy

        self.monitors[alg] = monitor

        if alg.environment not in self.monitors:
            self.monitors[alg.environment] = EnvMonitor()

        # init optimization time control
        monitor.optimize_start = time.time()

    def after_optimize(self, alg):
        """Catch data after optimization run."""
        monitor = self.monitors[alg]
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

            policy.setParameter(parameter)

            # compute trace
            trace = alg.environment._rollout(policy)
            monitor.traces.append(trace)

            # compute total reward
            reward = sum([t[2] for t in trace])
            monitor.rewards.append(reward)

    def before_step(self, alg):
        # count the number of rollouts for each step
        self.monitors[alg.environment].rollout_cnt = 0

        if self.verbose > 1:
            logger.info('Computing step %d for %s...', self.step_cnts[alg],
                        str(alg))

        # place holding code to make sure we see something at this stage
        monitor = self.monitors[alg]
        n = monitor.step_cnt
        parameter = monitor.policy.parameter

        if n == 0:
            self.t = time.time()
        elif n % 100 == 0:
            print("Run: " + str(n) + "  \tParameter: \t" + str(parameter)
                  + "\n\t\tGradient: \t" + str(alg.grad))
            now = time.time()
            print("\t\tAverage Time: "
                  + "\t{:.2f}".format((now - self.t) / 100)
                  + "s/step")
            self.t = now

    def after_step(self, alg):
        monitor = self.monitors[alg]
        emonitor = self.monitors[alg.environment]

        monitor.step_cnt += 1

        # store the number of rollouts since before step
        monitor.rollout_cnts.append(emonitor.rollout_cnt)

        # retrieve information
        policy = monitor.policy
        parameter = policy.parameter

        # store information
        monitor.parameters.append(parameter)


class EnvMonitor(object):
    """Class to store environment tracking data."""
    def __init__(self):
        self.rollout_cnt = 0


class AlgMonitor(object):
    """Class used to store algorithm tracking data."""
    def __init__(self):
        self.optimize_start = 0
        self.optimize_time = 0

        self.step_cnt = 0
        self.rollout_cnts = []

        self.parameters = []
        self.traces = []
        self.rewards = []
