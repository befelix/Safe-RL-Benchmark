"""Markov Decision Process Implementations."""

import numpy as np

from SafeRLBench import EnvironmentBase
from SafeRLBench.spaces import DiscreteSpace


# TODO: Implementation, Docs
class MDP(EnvironmentBase):
    """Discrete Markov Decision Process Environment.

    Attributes
    ----------
    transitions :

    rewards :

    action_space :

    state_space :

    init_state :

    state :

    """

    def __init__(self, transitions, rewards, init_state=None, seed=None):
        """MDP initialization.

        Parameters
        ----------
        transitions : array
            Array holding transition matrix for each action. The dimension of
            the state and action spaces will be deduced from this array.
        rewards : array
            Array holding the reward matrix for each action. It needs to comply
            with the dimensions deduced from the transitions array.
        """
        self.transitions = transitions
        self.rewards = rewards

        # determine state and action space
        self.action_space = DiscreteSpace(len(transitions))
        self.state_space = DiscreteSpace(transitions[0].shape[0])

        # setup current state and store the initial state for reset
        self.init_state = init_state
        self.state = init_state

    def _update(self, action):
        prev_state = self.state
        # choose next state
        self.state = np.random.choice(np.arange(self.action_space.dimension),
                                      p=self.transitions[action][self.state])
        # determine reward
        reward = self.rewards[action][prev_state, self.state]

        return action, self.state, reward

    def _reset(self):
        self.state = self.init_state
