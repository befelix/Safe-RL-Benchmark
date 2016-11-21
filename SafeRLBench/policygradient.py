from __future__ import division, print_function, absolute_import

import time

import numpy as np
from numpy.random import rand
from numpy.linalg import norm, solve

__all__ = ('PolicyGradient', 'PolicyGradientEstimator')


class PolicyGradient(object):
    def __init__(self, environment, estimator='reinforce',
                 max_it=200, eps=0.001, est_eps=0.001,
                 parameter_domain=np.array([0, 1]), rate=1):
        self.environment = environment
        self.state_dim = environment.state.shape[0]
        self.par_dim = self.state_dim + 1

        self.parameter_domain = parameter_domain

        self.parameters = []
        self.rewards = []

        self.rate = rate
        self.eps = eps
        self.max_it = max_it

        self.best_reward = -float("inf")
        self.best_parameter = None

        if isinstance(estimator, str):
            estimator = estimators[estimator]
        elif issubclass(estimator, PolicyGradientEstimator):
            pass
        else:
            raise ImportError('Invalid Estimator')

        self.estimator = estimator(environment, max_it, est_eps, rate)

    def optimize(self, policy):
        """
        Parameters:
        """
        estimator = self.estimator

        parameter = self._initialize_parameters(policy)

        converged = False

        print("Start " + estimator.name + " optimization:")
        print("Initial Parameters: " + str(parameter))

        for n in range(self.max_it):
            self.parameters.append(np.copy(parameter))
            grad, trace = estimator(policy, parameter)

            # store best result
            cummulative_reward = sum([x[2] for x in trace])
            self.rewards.append(cummulative_reward)

            if (cummulative_reward > self.best_reward):
                self.best_reward = cummulative_reward
                self.best_parameter = np.copy(parameter)

            # print once in a while for debugging
            if n % 100 == 0:
                print("Run: " + str(n) + "  \tParameter: \t" + str(parameter)
                      + "\tReward: " + str(cummulative_reward)
                      + "\n\t\tGradient: \t" + str(grad))
                if (n == 0):
                    t = time.time()
                else:
                    now = time.time()
                    print("\t\tAverage Time: "
                          + "\t{:.2f}".format((now - t) / 100)
                          + "s/step")
                    t = now

            # stop when gradient converges
            if norm(grad) < self.eps:
                converged = True
                break

            # update parameter
            parameter += self.rate * grad

        print("Finished")

        return (parameter, converged)

    def reset(self):
        self.parameters = []
        self.rewards = []

        self.best_reward = -float("inf")
        self.best_parameter = None
        self.best_goal = False

    def _initialize_parameters(self, policy):

        pard = self.parameter_domain

        while True:
            parameter = rand(self.par_dim) * (pard[1] - pard[0]) + pard[0]

            grad, _ = self.estimator(policy, parameter)

            if (norm(grad) >= self.eps):
                return (parameter)


class PolicyGradientEstimator(object):

    name = 'Policy Gradient'

    def __init__(self, environment, max_it=200, eps=0.001, rate=1):
        self.environment = environment
        self.state_dim = environment.state.shape[0]
        self.par_dim = self.state_dim + 1

        self.rate = rate
        self.eps = eps
        self.max_it = max_it

    def __call__(self, policy, parameter):
        return self._estimate_gradient(policy, parameter)


class ForwardFDEstimator(PolicyGradientEstimator):

    name = 'Forward Finite Differences'

    def __init__(self, environment, max_it=200,
                 eps=0.001, rate=1, var=0.5):
        super().__init__(environment, max_it, eps, rate)
        self.var = var

    def _estimate_gradient(self, policy, parameter):
        env = self.environment
        var = self.var
        # using forward differences
        policy.setParameter(parameter)
        trace = env.rollout(policy)
        Jref = sum([x[2] for x in trace]) / len(trace)

        dJ = np.zeros((2 * self.par_dim))
        dV = np.append(np.eye(self.par_dim), -np.eye(self.par_dim), axis=0)
        dV *= var

        for n in range(self.par_dim):
            variation = dV[n]

            policy.setParameter(parameter + variation)
            trace_n = env.rollout(policy)

            Jn = sum([x[2] for x in trace]) / len(trace_n)

            dJ[n] = Jref - Jn

        grad = solve(dV.T.dot(dV), dV.T.dot(dJ))

        return grad, trace


class CentralFDEstimator(PolicyGradientEstimator):

    name = 'Central Finite Differences'

    def __init__(self, environment, max_it=200,
                 eps=0.001, rate=1, var=0.5):
        super().__init__(environment, max_it, eps, rate)
        self.var = var

    def _estimate_gradient(self, policy, parameter):
        env = self.environment

        policy.setParameter(parameter)
        trace = env.rollout(policy)

        dJ = np.zeros((self.par_dim))
        dV = np.eye(self.par_dim) * self.var / 2

        for n in range(self.par_dim):
            variation = dV[n]

            policy.setParameter(parameter + variation)
            trace_n = env.rollout(policy)

            policy.setParameter(parameter - variation)
            trace_n_ref = env.rollout(policy)

            Jn = sum([x[2] for x in trace_n]) / len(trace_n)
            Jn_ref = sum([x[2] for x in trace_n_ref]) / len(trace_n_ref)

            dJ[n] = Jn - Jn_ref

        grad = solve(dV.T.dot(dV), dV.T.dot(dJ))

        return grad, trace


class ReinforceEstimator(PolicyGradientEstimator):

    name = 'Reinforce'

    def __init__(self, environment, max_it=200,
                 eps=0.001, rate=1, lam=0.5):
        super().__init__(environment, max_it, eps, rate)
        self.lam = lam

    def _estimate_gradient(self, policy, parameter):
        env = self.environment
        par_shape = parameter.shape
        max_it = self.max_it

        policy.setParameter(parameter)

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

            log_grad_sum = sum(list(map(policy.log_grad, states, actions)))

            b_div_n = log_grad_sum**2
            b_nom_n = b_div_n * rewards_sum

            b_div += b_div_n
            b_nom += b_nom_n

            b = b_nom / b_div
            grad_n = log_grad_sum * (rewards_sum - b)

            grads += grad_n

            grad_old = grad
            grad = grads / (n + 1)

            if (n > 2 and norm(grad_old - grad) < self.eps):
                return grad, trace

        print("Gradient did not converge!")
        return grad, trace


class GPOMDPEstimator(PolicyGradientEstimator):

    name = 'GPOMDP'

    def __init__(self, environment, max_it=200,
                 eps=0.001, rate=1, lam=0.5):
        super().__init__(environment, max_it, eps, rate)
        self.lam = lam

    def _estimate_gradient(self, policy, parameter):
        env = self.environment
        H = env.horizon
        shape = policy.parameter_shape

        b_nom = np.zeros((H, shape))
        b_div = np.zeros((H, shape))
        b = np.zeros((H, shape))
        grad = np.zeros(shape)

        lam = self.lam

        policy.setParameter(parameter)

        for n in range(self.max_it):
            trace = env.rollout(policy)
            b_n = np.zeros((H, shape))

            for k, state in enumerate(trace):
                update = policy.log_grad(state[1], state[0])
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
                update += policy.log_grad(state[1], state[0])
                grad_update += update * (-b[k] + state[2] * lam**k)

            if (n > 2 and norm(grad_update / (n + 1)) < self.eps):
                grad /= (n + 1)
                return grad, trace
            grad += np.nan_to_num(grad_update)

        print("Gradient did not converge!")

        grad /= n + 1
        return grad, trace


'''
Dictionary for resolving estimator strings
'''
estimators = {
    'forward_fd': ForwardFDEstimator,
    'central_fd': CentralFDEstimator,
    'reinforce': ReinforceEstimator,
    'gpomdp': GPOMDPEstimator
}
