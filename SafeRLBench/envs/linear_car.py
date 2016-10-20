import numpy as np
from numpy.linalg import norm

__all__=["LinearCar"]

def LinearCar(object):
    def __init__(self, initial_state=np.array([[0],[0]]), 
                       goal_position=np.array([1]),
                       step = 0.1, eps=0.1):
        self.initial_state    = initial_state
        self.state            = initial_state
        self.goal_position    = goal_position
        self.eps              = eps
        self.step             = step

        self.dim = initial_state.shape[1]

    def update(self, action):
        self.state[1] += step * action
        self.state[0] += self.state[1]

        return(action, self.state, self._reward(), self._achieved())

    def reset(self):
        self.state = self.initial_state

    def _reward(self):
        return -norm(self.state[0] - self.goal_position)

    def _achieved(self):
        return ((self._reward() + norm(self.state[1])) < self.eps)
