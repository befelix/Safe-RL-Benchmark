"""SafeOpt wrapper."""

from SafeRLBench import AlgorithmBase

from numpy import mean

import logging

logger = logging.getLogger(__name__)

try:
    import safeopt
except:
    logger.warning("SafeOpt is not installed.")


class SafeOpt(AlgorithmBase):
    """
    Wrap SafeOpt algorithm.

    Attributes
    ----------
    environment :
        environmet to be optimized.
    policy :
        policy to be optimized.
    max_it :
        maximal number of iterations before we abort.
    avg_reward : integer
        average reward at which the optimization will be finished.
    window : integer
        window for the average reward
    """

    def __init__(self,
                 environment, policy, max_it, avg_reward, window,
                 gp, parameter_set, fmin,
                 lipschitz=None, beta=3.0, num_contexts=0, threshold=0,
                 scaling='auto'):
        super(SafeOpt, self).__init__(environment, policy, max_it)
        self.gp_opt = safeopt(gp, parameter_set, fmin, lipschitz, beta,
                              num_contexts, threshold, scaling)
        self.avg_reward = avg_reward
        self.window = window
        self.rewards = []

    def _initialize(self):
        pass

    def _step(self):
        parameters = self.gp_opt.optimize()
        self.policy.parameters = parameters
        trace = self.environment.rollout(self.policy)
        reward = sum([t[2] for t in trace])
        self.gp_opt.add_new_data_point(parameters, reward)
        self.rewards.append(reward)

    def _is_finished(self):
        if ((len(self.rewards) < self.window
             and len(self.rewards > self.window / 2)
             and mean(self.rewards) > self.avg_reward)
            or (mean(self.rewards[len(self.rewards - self.window):-1]
                > self.avg_reward))):
            return True
        else:
            return False
