"""Markov Decision Process Implementations."""

from SafeRLBench import EnvironmentBase
from SafeRLBench.spaces import DiscreteSpace


class MDP(EnvironmentBase):
    """Discrete Markov Decision Process Environment.

    Attributes
    ----------

    """

    def __init__(self, transitions, rewards):
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

    def _update(self, action):
        pass

    def _reset(self):
        pass
