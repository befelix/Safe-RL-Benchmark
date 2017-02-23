"""Linear Policy Class."""

from SafeRLBench import Policy, ProbPolicy
from SafeRLBench.spaces import BoundedSpace

import numpy as np


class LinearPolicy(Policy):
    """
    Policy implementing a linear mapping from state to action space.

    Attributes
    ----------
    d_state : positive integer
        Dimension of the state space.
    d_action : positive integer
        Dimension of the action space
    parameters : nd-array
        Array containing initial parameters.
    initialized : boolean
        Boolean indicating if parameters have been initialized.
    biased : boolean
        Flag indicating if the policy is supposed to be biased or not.
    """

    def __init__(self, d_state, d_action, par=None, biased=True):
        """
        Initialize.

        Parameters
        ----------
        d_state : positive integer
            Dimension of the state space.
        d_action : positive integer
            Dimension of the action space
        par : ndarray
            Array containing initial parameters. If there is a constant bias,
            the array needs to be flat with shape (d_state * d_action + 1,).
            Otherwise it may either have shape (d_action, d_state) or
            (d_state * d_action,)
        biased : boolean
            Flag indicating if the policy is supposed to be biased or not.
        """
        assert(d_state > 0 and d_action > 0)
        self.d_state = d_state
        self.d_action = d_action

        self.par_dim = d_state * d_action
        self._par_space = None

        self.initialized = False

        if par is not None:
            self.parameters = par
        else:
            # make sure some fields exist.
            self._parameters = None
            self.biased = biased
            self._bias = 0
            self._par = None

    def map(self, state):
        """
        Map a state to an action.

        Parameters
        ----------
        state : array-like
            Element of state space.

        Return
        ------
        Element of action space.
        """
        if self.d_action == 1:
            ret = self._parameters.dot(state).item() + self._bias
        else:
            ret = self._parameters.dot(state) + self._bias
        return ret

    # private copy of map
    __map = map

    @property
    def parameters(self):
        """
        Property to access parameters.

        The property returns the same representation as used when set.
        If the mapping contains a bias, then the input needs to be a ndarray
        with shape (d_action * d_state + 1,) otherwise it may either be a
        (d_action, d_state) or (d_action * d_state,) shaped array
        """
        if not self.initialized:
            raise NameError('Policy parameters not initialized yet.')
        return self._par

    @parameters.setter
    def parameters(self, par):
        par = par.copy()
        shape = par.shape

        if not self.initialized:
            if (shape == (self.d_action, self.d_state)
                    or shape == (self.par_dim,)):
                self.biased = False
                self._bias = 0
            elif shape == (self.par_dim + 1,):
                self.biased = True
            else:
                raise ValueError("Parameters with shape %s invalid.",
                                 str(shape))

            self.initialized = True

        # store parameter in original representation.
        self._par = par

        if not self.biased:
            self._parameters = par
        else:
            self._bias = par[-1]
            self._parameters = par[0:-1].reshape((self.d_action, self.d_state))

    @property
    def parameter_space(self):
        if self._par_space is None:
            if self.biased:
                shape = (self.par_dim + 1,)
            else:
                shape = (self.par_dim,)
            self._par_space = BoundedSpace(0, 1, shape)

        return self._par_space

    @parameter_space.setter
    def parameter_space(self, par_space):
        self._par_space = par_space


class DiscreteLinearPolicy(LinearPolicy):
    """LinearPolicy on a descrete action space of {-1, 0, 1}^d"""

    def map(self, state):
        cont_action = super(DiscreteLinearPolicy, self).map(state)
        if self.d_action == 1:
            if (cont_action > 0):
                action = 1
            else:
                action = 0
        else:
            action = np.zeros(cont_action.shape, dtype=int)
            action[cont_action > 0] += 1
        return action


class NoisyLinearPolicy(LinearPolicy, ProbPolicy):
    """
    Policy implementing a linear mapping from state to action space with noise.

    Attributes
    ----------
    d_state : positive integer
        Dimension of the state space.
    d_action : positive integer
        Dimension of the action space
    sigma : double
        Sigma for gaussian noise
    parameters : nd-array
        Array containing initial parameters.
    initialized : boolean
        Boolean indicating if parameters have been initialized.
    biased : boolean
        Flag indicating if the policy is supposed to be biased or not.
    """

    def __init__(self, d_state, d_action, sigma, par=None, biased=False):
        """
        Initialize Noisy Linear Policy.

        Parameters
        ----------
        d_state : positive integer
            Dimension of the state space.
        d_action : positive integer
            Dimension of the action space
        sigma : double
            Sigma for gaussian noise
        par : ndarray
            Array containing initial parameters. If there is a constant bias,
            the array needs to be flat with shape (d_state * d_action + 1,).
            Otherwise it may either have shape (d_action, d_state) or
            (d_state * d_action,)
        biased : boolean
            Flag indicating if the policy is supposed to be biased or not.
        """
        assert(d_state > 0 and d_action > 0)

        self.sigma = sigma

        super(NoisyLinearPolicy, self).__init__(d_state, d_action, par, biased)

    def map(self, state):
        """
        Map a state to an action.

        Parameters
        ----------
        state : array-like
            Element of state space.

        Return
        ------
        Element of action space.
        """
        noise = np.random.normal(0, self.sigma)
        return super(NoisyLinearPolicy, self).map(state) + noise

    def grad_log_prob(self, state, action):
        """Compute the gradient of the logarithm of the probability dist."""
        noise = action - super(NoisyLinearPolicy, self).map(state)
        return - 2 * noise * self.parameters / self.sigma**2
