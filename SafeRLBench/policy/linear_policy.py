"""Linear Policy Class."""

from SafeRLBench import Policy, ProbPolicy

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
    """

    def __init__(self, d_state, d_action, par=None):
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
        """
        assert(d_state > 0 and d_action > 0)
        self.d_state = d_state
        self.d_action = d_action

        self.par_dim = d_state * d_action

        self.initialized = False

        if par is not None:
            self.parameters = par
        else:
            # make sure some fields exist.
            self._parameters = None
            self._bias = False
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
        return self._parameters.dot(state).item() + self._bias

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
                self._biased = False
                self._bias = 0
            elif shape == (self.par_dim + 1,):
                self._biased = True
            else:
                raise ValueError("Parameters with shape %s invalid.",
                                 str(shape))

            self.initialized = True

        # store parameter in original representation.
        self._par = par

        if not self._biased:
            self._parameters = par
        else:
            self._bias = par[-1]
            self._parameters = par[0:-1].reshape((self.d_action, self.d_state))


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
    """

    def __init__(self, d_state, d_action, sigma, par=None):
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
        """
        assert(d_state > 0 and d_action > 0)

        self.sigma = sigma

        super(NoisyLinearPolicy, self).__init__(d_state, d_action, par)

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
