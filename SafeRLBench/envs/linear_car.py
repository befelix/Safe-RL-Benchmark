import numpy as np
from numpy.linalg import norm

__all__=["LinearCar"]

class LinearCar(object):
    def __init__(self, initial_state=np.array([[0.],[0.]]), 
                       goal_position=np.array([1.]),
                       step = 0.01, eps=0.01, ):
        self.initial_state    = initial_state
        self.state            = np.copy(initial_state)
        self.goal_position    = goal_position
        self.eps              = eps
        self.step             = step

        self.dim = initial_state.shape[1]

    def update(self, action):
        action = max(min(action,1.),-1.)

        self.state[1] += self.step * action
        self.state[0] += self.state[1]

        return(action, self.state, self._reward(), self._achieved())

    def reset(self):
        self.state = np.copy(self.initial_state)

    def _reward(self):
        return -norm(self.state[0] - self.goal_position)-norm(self.state[1])

    def _achieved(self):
        return (-self._reward() < self.eps)
