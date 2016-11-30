"""Module implements Baseclasses."""

from __future__ import division, print_function, absolute_import

from SafeRLBench import config

__all__ = ('EnvironmentBase', 'Space')


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

    def __init__(self, state_space, action_space, horizon=0):
        self.state_space = state_space
        self.action_space = action_space
        self.horizon = horizon

    # retrieve global monitor
    @property
    def monitor(self):
        if not hasattr(self, '_monitor'):
            self._monitor = config.monitor
        return self._monitor

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

        Parameters
        ----------
        action: array-like
            Element of action_space

        Returns
        -------
        Tuple: (action, state, reward):
            action : array-like
                element of action space as it has been applied in update
            state : array-like
                element of state_space which is the resulting state after
                applying action
            reward : float
                reward for resulting state
        """
        self.monitor.before_update(self)
        t = self._update(action)
        self.monitor.after_update(self)
        return t

    def reset(self):
        """
        Wrap subclass implementation.

        Calls _reset() implementation of subclass.
        Supports addition of monitoring/benchmarking code.
        """
        self.monitor.before_reset(self)
        self._reset()
        self.monitor.after_reset(self)

    def rollout(self, policy):
        """
        Wrap subclass implementation.

        Calls _rollout(policy) implementation of subclass.
        Supports addition of monitoring/benchmarking code.

        Parameters
        ----------
        Policy : callable
            Maps element of state_space to element of action_space

        Returns
        -------
        trace : list of (action, state, reward)-tuples
        """
        self.monitor.before_rollout(self)
        trace = self._rollout(policy)
        self.monitor.after_rollout(self)
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


class AlgorithmBase(object):
    """
    Baseclass for any algorithm.

    This baseclass features monitoring capabilities for algorithm
    implementations. It is supposed to a uniform interface for any algorithm
    part of the algo module.

    Any subclass must overwrite:
        * _initialize(policy)
        * _step(policy)
        * _isFinished()

    Any subclass may overwrite:
        * _optimize(policy)

    In case one does overwrite _optimize, the functions _initialize(),
    _step(parameter), _isFinished() may just pass unless they are used.

    Attributes
    ----------
    max_it : int
        Maximum number of iterations
    monitor

    Methods
    -------
    optimize(policy)
        Optimize a policy with respective algorithm.
    initialize(policy)
        Initialize policy parameter.
    step(policy)
        Update policy parameters.
    isFinished()
        Return true when algorithm is finished.

    Notes
    -----
    Specification of the private functions.

    _initialize(self, policy):
        Return initial parameter for policy.
    _step(policy):
        Update policy parameter.
    _isFinished():
        Return True when algorithm is supposed to finish.
    """
    def __init__(self, max_it):
        self.max_it = max_it

    @property
    def monitor(self):
        """Lazily retrieve monitor to track execution."""
        if not hasattr(self, '_monitor'):
            self._monitor = config.monitor
        return self._monitor

    # Have to be overwritten.
    def _initialize(self, policy):
        """Return initial parameter."""
        raise NotImplementedError

    def _step(self, policy):
        """Update parameter of policy."""
        raise NotImplementedError

    def _isFinished(self):
        raise NotImplementedError

    # May be overwritten
    def _optimize(self, policy):
        self.initialize(policy)

        for n in range(self.max_it):
            self.step(policy)
            if self.isFinished():
                break

    def optimize(self, policy):
        """
        Optimize policy parameter.

        Wraps subclass implementation in _optimize(policy).

        Parameters
        ----------
        policy: PolicyBase subclass
        """
        self.monitor.before_optimize(self, policy)
        self._optimize(policy)
        self.monitor.after_optimize(self)

    def initialize(self, policy):
        """
        Initialize policy parameter.

        Wraps subclass implementation in _initialize(policy)

        Parameters
        ----------
        policy: PolicyBase subclass
        """
        parameter = self._initialize(policy)
        policy.setParameter(parameter)

    def step(self, policy):
        """
        Update policy parameter.

        Wraps subclass implementation in _step(policy).

        Parameters
        ----------
        policy: PolicyBase subclass
        """
        self.monitor.before_step(self)
        self._step(policy)
        self.monitor.after_step(self)

    def isFinished(self):
        """
        Return True when algorithm is supposed to finish.

        Wraps subclass implementation in _isFinished().
        """
        stop = self._isFinished()
        return stop
