"""Module implements Baseclasses."""

from __future__ import division, print_function, absolute_import

from SafeRLBench import config

__all__ = ('EnvironmentBase', 'Space')


class EnvironmentBase(object):
    """
    Environment Base Class.

    This base class defines and implements an interface to any environment
    implementation part of the environment module. Subclasses inheriting
    from EnvironmentBase need to make sure they meet the requirements below.

    Any subclass must implement:
        * _update(action)
        * _reset()

    Any subclass might override:
        * _rollout(policy)

    Initialization:
        Make sure super().__init__ is called in any case as there may be
        additions to the interface over time.

    Attributes
    ----------
    state_space :
    action_space :
    horizon :
        Maximum number of iterations until rollout will stop.
    monitor :
        Lazily retrieve monitor instance as soon as needed.

    Methods
    -------
    rollout(policy)
        Perform a rollout according to the actions selected by policy.
    update(action)
        Update the environment state according to the action.
    reset()
        Reset the environment to the initial state.

    Notes
    -----
    When overwriting _rollout(policy) use the provided interface functions
    and do not directly call the private implementation.
    """

    def __init__(self, state_space, action_space, horizon=0):
        self.state_space = state_space
        self.action_space = action_space
        self.horizon = horizon

    # retrieve global monitor
    @property
    def monitor(self):
        """
        Lazily retrieve monitor instance as soon as needed.

        The monitor is globally defined and allows dynamic tracking of method
        calls in the environment and algorithm classes.
        """
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
        Update the environment state according to the action.

        Wraps the subclass implementation _update(action) providing
        monitoring capabilities.

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
        Reset the environment to initial state.

        Reset wraps the subclass implementation _reset() providing monitoring
        capabilities.
        """
        self.monitor.before_reset(self)
        self._reset()
        self.monitor.after_reset(self)

    def rollout(self, policy):
        """
        Perform a rollout according to the actions selected by policy.

        Wraps the implementation _rollout(policy) providing monitoring
        capabilities.

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

    Methods
    -------
    contains(x)
        Check if x is an element of space.
    element
        Return arbitray element in space.
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

    This baseclass defines a uniform interface for any algorithm part of
    the algorithm module SafeRLBench.algo.
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
    environment :
        Environment we want to optimize on
    policy :
        Policy to be optimized
    max_it : int
        Maximum number of iterations
    monitor :
        Lazily retrieve monitor to track execution

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
    def __init__(self, environment, policy, max_it):
        self.environment = environment
        self.policy = policy
        self.max_it = max_it

    @property
    def monitor(self):
        """Lazily retrieve monitor to track execution."""
        if not hasattr(self, '_monitor'):
            self._monitor = config.monitor
        return self._monitor

    # Have to be overwritten.
    def _initialize(self):
        raise NotImplementedError

    def _step(self):
        raise NotImplementedError

    def _isFinished(self):
        raise NotImplementedError

    # May be overwritten
    def _optimize(self):
        self.initialize()

        for n in range(self.max_it):
            self.step()
            if self.isFinished():
                break

    def optimize(self):
        """
        Optimize policy parameter.

        Wraps subclass implementation in _optimize(policy).

        Parameters
        ----------
        policy: PolicyBase subclass
        """
        self.monitor.before_optimize(self)
        self._optimize()
        self.monitor.after_optimize(self)

    def initialize(self):
        """
        Initialize policy parameter.

        Wraps subclass implementation in _initialize(policy)

        Parameters
        ----------
        policy: PolicyBase subclass
        """
        parameter = self._initialize()
        self.policy.setParameter(parameter)

    def step(self):
        """
        Update policy parameter.

        Wraps subclass implementation in _step(policy).

        Parameters
        ----------
        policy: PolicyBase subclass
        """
        self.monitor.before_step(self)
        self._step()
        self.monitor.after_step(self)

    def isFinished(self):
        """
        Return True when algorithm is supposed to finish.

        Wraps subclass implementation in _isFinished().
        """
        stop = self._isFinished()
        return stop
