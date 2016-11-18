import numpy as np


class EnvironmentBase(object):
    """
    Environment Base Class
    """

    def __init__(self, state_space, action_space, horizon):
        self.state_space = state_space
        self.action_space = action_space
        self.horizon = horizon

    def update(self, action):
        """
        Wraps environments _update(action) implementation
        Supports addition of monitoring/benchmarking code

        Parameters:
        -----------
        action: array-like
            Element of action_space

        Return:
        -------
        Tuple: (action, state, reward):
            action:
                element of action space as it has been applied in update
            state:
                element of state_space which is the resulting state after
                applying action
            reward:
                reward for resulting state
        """
        ret = self._update(action)
        return ret

    def reset(self):
        """
        Wraps environments _reset() implementation
        Supports notification of reset operations to monitoring/benchmarking
        facilities.
        """
        self._reset()

    def rollout(self, policy):
        """
        Wraps environments _rollout(policy) implementation.
        Supports monitoring the _rollout operation.

        Parameters:
        -----------
        Policy: callable
            Maps element of state_space to element of action_space

        Returns:
        --------
        trace: list of (action, state, reward)-tuples
        """
        trace = self._rollout(policy)

    # Override in subclasses if necessary
    def _rollout(self, policy):
        trace = []
        for n in range(horizon):
            action = policy(self.state)
            trace.append(self.update(action))
        return trace

    # Implement in subclasses:
    def _update(self, action):
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError
