"""Policy Gradient implementations."""

from SafeRLBench import AlgorithmBase
from SafeRLBench.spaces import BoundedSpace

import numpy as np
from numpy.linalg import solve, norm

from abc import ABCMeta, abstractmethod
from six import add_metaclass

import logging

logger = logging.getLogger(__name__)


class PolicyGradient(AlgorithmBase):
    """
    Implementing many policy gradient methods.

    Attributes
    ----------
    environment :
        environment we want to run the optimization
    policy :
        the policy to be optimized
    """

    def __init__(self,
                 environment, policy, estimator='reinforce',
                 max_it=1000, eps=0.0001, est_eps=0.001,
                 parameter_space=BoundedSpace(0, 1, (3,)),
                 rate=1, var=0.5):
        """Initialize PolicyGradient."""
        super(PolicyGradient, self).__init__(environment, policy, max_it)

        self.parameter_space = policy.parameter_space

        self.eps = eps
        self.rate = rate

        if isinstance(estimator, str):
            estimator = estimators[estimator]
        elif issubclass(estimator, PolicyGradientEstimator):
            pass
        else:
            raise ImportError('Invalid Estimator')

        self.estimator = estimator(environment, self.parameter_space, max_it,
                                   est_eps, rate, var)

    def _initialize(self):
        parameter = self.parameter_space.element()
        for n in range(10000):
            self.policy.parameters = parameter
            grad = self.estimator(self.policy)

            if (norm(grad) >= 1000 * self.eps):
                return parameter

            parameter = self.parameter_space.element()

        logger.error('Unable to find non-zero gradient.')
        return parameter

    def _step(self):
        grad = self.estimator(self.policy)

        parameter = self.policy.parameters

        self.policy.parameters = parameter + grad

        self.grad = grad

    def _is_finished(self):
        return (norm(self.grad) < self.eps)


@add_metaclass(ABCMeta)
class PolicyGradientEstimator(object):
    """Interface for Gradient Estimators."""

    name = 'Policy Gradient'

    def __init__(self, environment, parameter_space, max_it=200, eps=0.001,
                 rate=1):
        """Initialize."""
        self.environment = environment
        self.state_dim = environment.state.shape[0]
        self.par_dim = parameter_space.dimension

        self.rate = rate
        self.eps = eps
        self.max_it = max_it

    def __repr__(self):
        return self.__class__.__name__

    def __call__(self, policy):
        """Invoke _estimate_gradient(policy)."""
        return self._estimate_gradient(policy)

    @abstractmethod
    def _estimate_gradient(self, policy):
        pass


class ForwardFDEstimator(PolicyGradientEstimator):
    """Forward Finite Differences Gradient Estimator."""

    name = 'Forward Finite Differences'

    def __init__(self, environment, parameter_space, max_it=200, eps=0.001,
                 rate=1, var=1):
        """Initialize."""
        super(ForwardFDEstimator, self).__init__(environment, parameter_space,
                                                 max_it, eps, rate)
        self.var = var

    def _estimate_gradient(self, policy):
        env = self.environment
        var = self.var
        # store current policy parameter
        parameter = policy.parameters
        par_dim = policy.parameter_space.dimension

        # using forward differences
        trace = env.rollout(policy)
        j_ref = sum([x[2] for x in trace]) / len(trace)

        dj = np.zeros((2 * par_dim))
        dv = np.append(np.eye(par_dim), -np.eye(par_dim), axis=0)
        dv *= var

        for n in range(par_dim):
            variation = dv[n]

            policy.parameters = parameter + variation
            trace_n = env.rollout(policy)

            jn = sum([x[2] for x in trace]) / len(trace_n)

            dj[n] = j_ref - jn

        grad = solve(dv.T.dot(dv), dv.T.dot(dj))

        # reset current policy parameter
        policy.parameters = parameter

        return grad


class CentralFDEstimator(PolicyGradientEstimator):
    """Central Finite Differences Gradient Estimator."""

    name = 'Central Finite Differences'

    def __init__(self, environment, parameter_space, max_it=200, eps=0.001,
                 rate=1, var=1):
        """Initialize."""
        super(CentralFDEstimator, self).__init__(environment, parameter_space,
                                                 max_it, eps, rate)
        self.var = var

    def _estimate_gradient(self, policy):
        env = self.environment

        parameter = policy.parameters
        par_dim = policy.parameter_space.dimension

        dj = np.zeros((par_dim,))
        dv = np.eye(par_dim) * self.var / 2

        for n in range(par_dim):
            variation = dv[n]

            policy.parameters = parameter + variation
            trace_n = env.rollout(policy)

            policy.parameters = parameter - variation
            trace_n_ref = env.rollout(policy)

            jn = sum([x[2] for x in trace_n]) / len(trace_n)
            jn_ref = sum([x[2] for x in trace_n_ref]) / len(trace_n_ref)

            dj[n] = jn - jn_ref

        grad = solve(dv.T.dot(dv), dv.T.dot(dj))
        policy.parameters = parameter

        return grad


class ReinforceEstimator(PolicyGradientEstimator):
    """Reinforce Gradient Estimator."""

    name = 'Reinforce'

    def __init__(self, environment, parameter_space, max_it=200, eps=0.001,
                 rate=1, lam=0.5):
        """Initialize."""
        super(ReinforceEstimator, self).__init__(environment, parameter_space,
                                                 max_it, eps, rate)
        self.lam = lam

    def _estimate_gradient(self, policy):
        env = self.environment
        par_shape = policy.parameters.shape
        max_it = self.max_it

        b_div = np.zeros(par_shape)
        b_nom = np.zeros(par_shape)

        grads = np.zeros(par_shape)
        grad = np.zeros(par_shape)

        for n in range(max_it):
            trace = env.rollout(policy)

            lam = self.lam

            actions = [x[0] for x in trace]
            states = [x[1] for x in trace]

            rewards_sum = sum([x[2] * lam**k for k, x in enumerate(trace)])

            lg_sum = sum(list(map(policy.grad_log_prob, states, actions)))

            b_div_n = lg_sum**2
            b_nom_n = b_div_n * rewards_sum

            b_div += b_div_n
            b_nom += b_nom_n

            b = b_nom / b_div
            grad_n = lg_sum * (rewards_sum - b)

            grads += grad_n

            grad_old = grad
            grad = grads / (n + 1)

            if (n > 2 and norm(grad_old - grad) < self.eps):
                return grad

        logger.warning('ReinforceEstimator did not converge!'
                       + 'You may want to raise max_it.')
        return grad


class GPOMDPEstimator(PolicyGradientEstimator):
    """GPOMDP Gradient Estimator."""

    name = 'GPOMDP'

    def __init__(self, environment, parameter_space, max_it=200, eps=0.001,
                 rate=1, lam=0.5):
        """Initialize."""
        super(GPOMDPEstimator, self).__init__(environment, parameter_space,
                                              max_it, eps, rate)
        self.lam = lam

    def _estimate_gradient(self, policy):
        env = self.environment
        h = env.horizon
        shape = policy.parameters.shape

        b_nom = np.zeros((h, shape))
        b_div = np.zeros((h, shape))
        b = np.zeros((h, shape))
        grad = np.zeros(shape)

        lam = self.lam

        for n in range(self.max_it):
            trace = env.rollout(policy)
            b_n = np.zeros((h, shape))

            for k, state in enumerate(trace):
                update = policy.grad_log_prob(state[1], state[0])
                for j in range(k + 1):
                    b_n[j] += update

            fac = n / (n + 1)

            b_n = b_n**2
            b_div = fac * b_div + b_n / (n + 1)

            for k, state in enumerate(trace):
                b_nom[k] = fac * b_nom[k]
                b_nom[k] += b_n[k] * state[2] * lam**k / (n + 1)

            b = b_nom / b_div

            grad_update = np.zeros(shape)
            update = np.zeros(shape)
            for k, state in enumerate(trace):
                update += policy.grad_log_prob(state[1], state[0])
                grad_update += update * (-b[k] + state[2] * lam**k)

            if (n > 2 and norm(grad_update / (n + 1)) < self.eps):
                grad /= (n + 1)
                return grad
            grad += np.nan_to_num(grad_update)

        logger.warning('GPOMDP did not converge! '
                       + 'You may want to raise max_it.')
        grad /= n + 1
        return grad


"""Dictionary for resolving estimator strings."""
estimators = {
    'forward_fd': ForwardFDEstimator,
    'central_fd': CentralFDEstimator,
    'reinforce': ReinforceEstimator,
    'gpomdp': GPOMDPEstimator
}
