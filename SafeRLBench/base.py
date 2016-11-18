import numpy as np

__all__ = ['EnvironmentBase', 'Space']


class EnvironmentBase(object):
    """
    Environment Base Class

    The functions update, reset and rollout are wrappers for the deferred
    implementations _update, _reset and _rollout.

    Any subclass must implement the following functions:
    _update(self, action)
    _reset(self)

    Any subclass might override the following functions:
    _rollout(policy)

    Any subclass must initialize the following variables:
    state_space
    action_space
    horizon - unless _rollout(policy) gets overwritten
    """

    # initialize these
    state_space = None
    action_space = None
    horizon = 0

    # Implement in subclasses:
    # See update(self, action) for more information
    def _update(self, action):
        raise NotImplementedError

    # See reset(self) for more information
    def _reset(self):
        raise NotImplementedError

    # Override in subclasses if necessary
    def _rollout(self, policy):
        trace = []
        for n in range(horizon):
            action = policy(self.state)
            trace.append(self.update(action))
        return trace

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


class Space(object):

    def contains(self, x):
        """
        Check if x is an element of space.
        """
        raise NotImplementedError
