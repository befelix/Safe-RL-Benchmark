import logging
import time


class Monitor(object):
    """This class is used to track algorithms and environments."""

    def __init__(self):
        self.log = logging.getLogger(__name__)

        # rollout
        self.rollout_count = {}

        # optimize
        self._optimize_start = {}
        self.optimize_times = {}
        self.optimize_policy = {}

        # step
        self.parameters = {}
        self.traces = {}
        self.rewards = {}
        self.step_cnt = {}

    def before_update(self, env):
        pass

    def after_update(self, env):
        pass

    def before_rollout(self, env):
        if env not in self.rollout_count:
            self.rollout_count[env] = 0

    def after_rollout(self, env):
        self.rollout_count[env] += 1

    def before_reset(self, env):
        pass

    def after_reset(self, env):
        pass

    def before_optimize(self, alg, policy):
        # register algorithm
        self._optimize_start[alg] = time.time()
        self.optimize_policy[alg] = policy
        self.optimize_times[alg] = []
        self.step_cnt[alg] = 0
        self.parameters[alg] = []
        self.traces[alg] = []
        self.rewards[alg] = []

    def after_optimize(self, alg):
        # retrieve time of optimization
        optimize_end = time.time()
        optimize_time = optimize_end - self._optimize_start[alg]

        if self._optimize_start[alg] == 0:
            self.log.warning('Time measure for optimize corrupted')

        self._optimize_start[alg] = 0

        self.optimize_times[alg].append(optimize_time)

        # independently compute traces after optimization is finished
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
        n = self.step_cnt[alg]
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
        self.step_cnt[alg] += 1

        # retrieve information
        policy = self.optimize_policy[alg]
        parameter = policy.parameter

        # store information
        self.parameters[alg].append(parameter)
