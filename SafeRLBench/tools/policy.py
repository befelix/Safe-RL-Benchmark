import numpy as np

__all__ = ['Policy']


class Policy(object):

    def __init__(self, par_policy, parameter_shape, gradient=None, sigma=0.):

        self.parameter_shape = parameter_shape
        self.parameter = np.empty(parameter_shape)

        self.par_policy = par_policy
        self.policy = lambda x: 1

        self.sigma = sigma
        self.gradient = gradient

    def __call__(self, state):

        if (self.sigma == 0):
            noise = 0.
        else:
            noise = np.random.normal(0, self.sigma)

        return self.policy(state) + noise

    def log_grad(self, state, action):
        if self.gradient is not None:
            par = self.parameter
            sigma = self.sigma
            diff = action - self.policy(state)
            ret = 2 * diff * self.gradient(state, action, par) / sigma**2
            return ret
        else:
            print("Gradient not set!")
            return np.zeros(parameter_shape)

    def setParameter(self, parameter):
        self.parameter = parameter
        self.policy = self.par_policy(parameter)
