"""Module implements Baseclasses."""

from __future__ import division, print_function, absolute_import

__all__ = ['EnvironmentBase', 'Space']


class EnvironmentBase(object):
    """
    Environment Base Class.

    The methods update, reset and rollout are wrappers for the deferred
    implementations _update, _reset and _rollout.

    Any subclass must implement the following methods:
    _update(self, action)
    _reset(self)

    Any subclass might override the following methods:
    _rollout(policy)

    Any subclass must initialize the following variables:
    state_space
    action_space
    horizon - unless _rollout(policy) gets overwritten
    """

    # Initialize these
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
        self.reset()
        trace = []
        for n in range(self.horizon):
            action = policy(self.state)
            trace.append(self.update(action))
        return trace

    def update(self, action):
        """
        Wrap subclass implementation.

        This method calls the _update(action) implementation which
        has to be implemented in any subclass.
        Supports addition of monitoring/benchmarking code.

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
        Wrap subclass implementation.

        Calls _reset() implementation of subclass.
        Supports addition of monitoring/benchmarking code.
        """
        self._reset()

    def rollout(self, policy):
        """
        Wrap subclass implementation.

        Calls _rollout(policy) implementation of subclass.
        Supports addition of monitoring/benchmarking code.

        Parameters:
        -----------
        Policy: callable
            Maps element of state_space to element of action_space

        Returns:
        --------
        trace: list of (action, state, reward)-tuples
        """
        trace = self._rollout(policy)
        return trace


class Space(object):
    """
    Baseclass for Spaceobject.

    All methods have to be implemented in any subclass.
    """

    def contains(self, x):
        """Check if x is an element of space."""
        raise NotImplementedError

    def element(self):
        """Return an arbitrary element in space for unit testing."""
        raise NotImplementedError
