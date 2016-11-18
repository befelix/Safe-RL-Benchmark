from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.linalg import norm

from SafeRLBench import EnvironmentBase
from SafeRLBench.spaces import RdSpace


class LinearCar(EnvironmentBase):
    def __init__(self,
                 state_space=RdSpace((2, 1)), action_space=RdSpace((1,)),
                 state=np.array([[0.], [0.]]), goal=np.array([[1.], [0.]]),
                 step=0.01, eps=0.01, horizon=100):
        # Initialize EnivronmentBase Parameters
        self.state_space = state_space
        self.action_space = action_space
        self.horizon = horizon

        # Initialize State
        self.initial_state = state
        self.state = np.copy(state)

        # Initialize Environment Parameters
        self.goal = goal
        self.eps = eps
        self.step = step

    def _update(self, action):
        one = np.ones(self.action_space.shape)
        action = np.maximum(np.minimum(action, one), -one)

        self.state[1] += self.step * action
        self.state[0] += self.state[1]

        return (action, self.state, self._reward())

    def _reset(self):
        self.state = np.copy(self.initial_state)

    def _reward(self):
        return -norm(self.state - self.goal)

    def _achieved(self):
        return (abs(self._reward()) < self.eps)
