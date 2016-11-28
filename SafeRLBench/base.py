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
    monitor = config.monitor

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
        self.monitor.before_update(self)
        ret = self._update(action)
        self.monitor.after_update(self)
        return ret

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

        Parameters:
        -----------
        Policy: callable
            Maps element of state_space to element of action_space

        Returns:
        --------
        trace: list of (action, state, reward)-tuples
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

    The optimize method wraps the _optimize implementation which default
    implementation makes use of initialize(policy), step(policy) and
    isFinished().
    These funtions are wrappers for the listed implementations below,
    supporting logging and monitoring of the algorithm.

    Any subclass must overwrite:
    _initialize(policy)
    _step(policy)
    _isFinished()

    Any subclass may overwrite:
    _optimize(policy)

    It might be infeasable to allow overwriting _optimize, so this will
    potentially change.
    In case one does overwrite _optimize, the functions _initialize(),
    _step(parameter), _isFinished() may just pass unless they are used.

    Requirements:
    _initialize(policy):
        Determine and set initial parameter for policy.
    _step(policy):
        Update policy parameter.
        Return current reward.
    _isFinished():
        Return True when algorithm is supposed to finish.
    """

    def __new__(cls, *args, **kwargs):
        """
        Create an instance of an algorithm.

        Initialize important tracking variables.
        """
        alg = super(AlgorithmBase, cls).__new__(cls)

        alg.parameters = []
        alg.rewards = []

        alg.best_reward = -float('inf')
        alg.best_parameter = None

    # Have to be overwritten.
    def _initialize(self, policy):
        raise NotImplementedError

    def _step(self, policy):
        raise NotImplementedError

    def _isFinished(self):
        raise NotImplementedError

    # May be overwritten
    def _optimize(self, policy):
        self.initialize(policy)
        stop = False
        while not stop:
            self.step(policy)
            stop = self.isFinished()

    def optimize(self, policy):
        """
        Optimize policy parameter.

        Wraps subclass implementation in _optimize(policy).

        Parameter:
        ----------
        policy: PolicyBase subclass
        """
        self._optimize(policy)

    def initialize(self, policy):
        """
        Initialize policy parameter.

        Wraps subclass implementation in _initialize(policy)

        Parameter:
        ----------
        policy: PolicyBase subclass
        """
        self._initialize(policy)

    def step(self, policy):
        """
        Update policy parameter.

        Wraps subclass implementation in _step(policy).

        Parameter:
        ----------
        policy: PolicyBase subclass
        """
        reward = self._step(policy)

        self.rewards.append(reward)
        self.parameters.append(policy.parameters)

        if reward > self.best_reward:
            self.best_reward = reward
            self.best_parameter = policy.parameter

    def isFinished(self):
        """
        Return True when algorithm is supposed to finish.

        Wraps subclass implementation in _isFinished().
        """
        self.isFinished()
