"""Linear Car."""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy import copy
from numpy.linalg import norm

from SafeRLBench import EnvironmentBase
from SafeRLBench.spaces import RdSpace


class LinearCar(EnvironmentBase):
    """Implementation of LinearCar Environment."""

    def __init__(self,
                 state_space=RdSpace((2, 1)), action_space=RdSpace((1,)),
                 state=np.array([[0.], [0.]]), goal=np.array([[1.], [0.]]),
                 step=0.01, eps=0.01, horizon=100):
        """
        Initialize EnvironmentBase parameters and specific parameters.

        Baseclass Parameters as in base.py.

        Parameters:
        -----------
        state: array-like
            Element of state_space. Specifies initial state.
        goal: array-like
            Element of state_space. Specifies goal state.
            The goal state should contain zero velocity, anything else does
            not make sense.
        step: double
        eps: double
            Reward at which we want to abort. If zero we do not abort at all.
        """
        # Initialize EnivronmentBase Parameters
        self.state_space = state_space
        self.action_space = action_space
        self.horizon = horizon

        # Initialize State
        self.initial_state = state
        self.state = copy(state)

        # Initialize Environment Parameters
        self.goal = goal
        self.eps = eps
        self.step = step

    def _update(self, action):
        one = np.ones(self.action_space.shape)
        action = np.maximum(np.minimum(action, one), -one)

        self.state[1] += self.step * action
        self.state[0] += self.state[1]

        return (action, copy(self.state), self._reward())

    def _reset(self):
        self.state = copy(self.initial_state)

    def _rollout(self, policy):
        self.reset()
        trace = []
        for n in range(self.horizon):
            action = policy(self.state)
            trace.append(self.update(action))
            if (self.eps != 0 and self._achieved()):
                return trace
        return trace

    def _reward(self):
        return -norm(self.state - self.goal)

    def _achieved(self):
        return (abs(self._reward()) < self.eps)
