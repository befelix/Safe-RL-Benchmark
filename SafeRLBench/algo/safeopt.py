"""SafeOpt Wrapper."""

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
    gp: GPy Gaussian process
        A Gaussian process which is initialized with safe, initial data points.
        If a list of GPs then the first one is the value, while all the
        other ones are safety constraints.
    parameter_set: 2d-array
        List of parameters
    fmin: list of floats
        Safety threshold for the function value. If multiple safety constraints
        are used this can also be a list of floats (the first one is always
        the one for the values, can be set to None if not wanted)
    lipschitz: list of floats
        The Lipschitz constant of the system, if None the GP confidence
        intervals are used directly.
    beta: float or callable
        A constant or a function of the time step that scales the confidence
        interval of the acquisition function.
    threshold: float or list of floats
        The algorithm will not try to expand any points that are below this
        threshold. This makes the algorithm stop expanding points eventually.
        If a list, this represents the stopping criterion for all the gps.
        This ignores the scaling factor.
    scaling: list of floats or "auto"
        A list used to scale the GP uncertainties to compensate for
        different input sizes. This should be set to the maximal variance of
        each kernel. You should probably leave this to "auto" unless your
        kernel is non-stationary.
    """

    def __init__(self,
                 environment, policy, max_it, avg_reward, window,
                 gp, parameter_set, fmin,
                 lipschitz=None, beta=3.0, num_contexts=0, threshold=0,
                 scaling='auto'):
        """Initialize Attributes."""
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
