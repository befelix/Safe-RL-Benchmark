import logging
import time

logger = logging.getLogger(__name__)


class Monitor(object):
    """This class is used to track algorithms and environments."""

    def __init__(self, verbose=0):
        self.verbose = verbose

        self.monitors = {}

        # setup default algorithm monitor
        self.alg_monitor = {
            # init tracking information
            'step_cnt': 0,
            'rollout_cnts': [],

            'parameters': [],
            'traces': [],
            'rewards': [],
        }

        # setup default environment monitor
        self.env_monitor = {
            '_rollout_cnt': 0
        }
        # rollout
        self._rollout_cnt = {}

        # optimize
        self._optimize_start = {}
        self.optimize_times = {}
        self.optimize_policy = {}

        # step
        self.parameters = {}
        self.traces = {}
        self.rewards = {}
        self.step_cnts = {}
        self.rollout_cnts = {}

    def before_update(self, env):
        pass

    def after_update(self, env):
        pass

    def before_rollout(self, env):
        if env not in self.monitors:
            self.monitors[env] = {'_rollout_cnt': 0}

    def after_rollout(self, env):
        self.monitors[env]['_rollout_cnt'] += 1

    def before_reset(self, env):
        pass

    def after_reset(self, env):
        pass

    def before_optimize(self, alg, policy):
        """Setup montitor for optimization run."""
        if self.verbose > 0:
            logger.info('Starting optimization of %s...', str(alg))

        # init monitor dict for algorithm
        monitor = self.alg_monitor.copy()
        monitor['policy'] = policy

        self.monitors[alg] = monitor

        if alg.environment not in self.monitors:
            self.monitors[alg.environment] = self.env_monitor.copy()

        # init optimization time control
        monitor['_optimize_start'] = time.time()

    def after_optimize(self, alg):
        """Catch data after optimization run."""
        monitor = self.monitors[alg]
        # retrieve time of optimization
        optimize_end = time.time()
        optimize_time = optimize_end - monitor['_optimize_start']

        if monitor['_optimize_start'] == 0:
            logger.warning('Time measure for optimize corrupted')

        monitor['_optimize_start'] = 0

        monitor['optimize_time'] = optimize_time

        # independently compute traces after optimization is finished
        if self.verbose > 0:
            logger.info('Computing traces for %s run...', str(alg))

        policy = monitor['policy']

        for parameter in monitor['parameters']:

            policy.setParameter(parameter)

            # compute trace
            trace = alg.environment._rollout(policy)
            monitor['traces'].append(trace)

            # compute total reward
            reward = sum([t[2] for t in trace])
            monitor['rewards'].append(reward)

    def before_step(self, alg):
        # count the number of rollouts for each step
        self.monitors[alg.environment]['rollout_cnt'] = 0

        if self.verbose > 1:
            logger.info('Computing step %d for %s...', self.step_cnts[alg],
                        str(alg))

        # place holding code to make sure we see something at this stage
        monitor = self.monitors[alg]
        n = monitor['step_cnt']
        parameter = monitor['policy'].parameter

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

        monitor['step_cnt'] += 1

        # store the number of rollouts since before step
        monitor['rollout_cnts'].append(emonitor['_rollout_cnt'])

        # retrieve information
        policy = monitor['policy']
        parameter = policy.parameter

        # store information
        monitor['parameters'].append(parameter)
