import numpy as np

__all__ = ['Policy']

class Policy(object):
    def __init__(self, par_policy, parameter_shape):

        self.parameter_shape = parameter_shape
        self.parameter = np.empty(parameter_shape)

        self.par_policy = par_policy
        self.policy = lambda x: 1

    def __call__(self, state):
        return self.policy(state)

    def setParameter(self, parameter):
        self.parameter = parameter
        self.policy = self.par_policy(parameter)
