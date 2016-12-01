import logging
import time

logger = logging.getLogger(__name__)


class Monitor(object):
    """
    This class is used to track algorithms and environments.

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
        self.verbose = verbose

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
        if env not in self._rollout_cnt:
            self._rollout_cnt[env] = 0

    def after_rollout(self, env):
        self._rollout_cnt[env] += 1

    def before_reset(self, env):
        pass

    def after_reset(self, env):
        pass

    def before_optimize(self, alg, policy):
        """Setup montitor for optimization run."""
        if self.verbose > 0:
            logger.info('Starting optimization of %s...', str(alg))

        # register algorithm
        # init policy
        self.optimize_policy[alg] = policy

        # init tracking information
        self.step_cnts[alg] = 0
        self.parameters[alg] = []
        self.traces[alg] = []
        self.rewards[alg] = []

        # init optimization time control
        self.optimize_times[alg] = []
        self._optimize_start[alg] = time.time()

    def after_optimize(self, alg):
        """Catch data after optimization run."""
        # retrieve time of optimization
        optimize_end = time.time()
        optimize_time = optimize_end - self._optimize_start[alg]

        if self._optimize_start[alg] == 0:
            logger.warning('Time measure for optimize corrupted')

        self._optimize_start[alg] = 0

        self.optimize_times[alg].append(optimize_time)

        # independently compute traces after optimization is finished
        if self.verbose > 0:
            logger.info('Computing traces for %s run...', str(alg))

        policy = self.optimize_policy[alg]

        for parameter in self.parameters[alg]:

            policy.setParameter(parameter)

            # compute trace
            trace = alg.environment._rollout(policy)
            self.traces[alg].append(trace)

            # compute total reward
            reward = sum([t[2] for t in trace])
            self.rewards[alg].append(reward)

    def before_step(self, alg):
        # count the number of rollouts for each step
        self._rollout_cnt[alg.environment] = 0

        if self.verbose > 1:
            logger.info('Computing step %d for %s...', self.step_cnts[alg],
                        str(alg))

        # place holding code to make sure we see something at this stage
        n = self.step_cnts[alg]
        parameter = self.optimize_policy[alg].parameter
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

        self.step_cnts[alg] += 1

        # store the number of rollouts since before step
        self.rollout_cnts[alg].append(self.rollout_cnt[alg.environment])

        # retrieve information
        policy = self.optimize_policy[alg]
        parameter = policy.parameter

        # store information
        self.parameters[alg].append(parameter)
