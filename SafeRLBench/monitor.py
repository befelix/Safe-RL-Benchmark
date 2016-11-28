import logging
import time


class Monitor(object):
    """This class is used to track algorithms and environments."""

    def __init__(self):
        self.log = logging.getLogger(__name__)

        # tracking the time
        # update
        self.update_start = {}
        self.update_times = {}

        # rollout
        self.rollout_start = {}
        self.rollout_times = {}

        # alg
        self.optimize_start = {}
        self.optimize_times = {}

    def before_update(self, env):

        # last call
        self.update_start[env] = time.time()

    def after_update(self, env):
        update_end = time.time()
        update_time = update_end - self.update_start[env]

        if self.update_start[env] == 0:
            self.log.warning('Time measure for update corrupted.')

        self.update_start[env] = 0

        if env not in self.update_times:
            self.update_times[env] = [update_time]
        else:
            self.update_times[env].append(update_time)

    def before_rollout(self, env):
        # last call
        self.rollout_start[env] = time.time()

    def after_rollout(self, env):
        rollout_end = time.time()
        rollout_time = rollout_end - self.rollout_start[env]

        if self.rollout_start[env] == 0:
            self.log.warning('Time measure for rollout corrupted.')

        self.rollout_start[env] = 0

        if env not in self.rollout_times:
            self.rollout_times[env] = [rollout_time]
        else:
            self.rollout_times[env].append(rollout_time)

    def before_reset(self, env):
        pass

    def after_reset(self, env):
        pass

    def before_optimize(self, alg):
        self.optimize_start[alg] = time.time()

    def after_optimize(self, alg):
        # retrieve time of optimization
        optimize_end = time.time()
        optimize_time = optimize_end - self.optimize_start[alg]

        if alg not in self.optimize_times:
            self.optimize_times[alg] = [optimize_time]
        else:
            self.optimize_times[alg].append(optimize_time)
