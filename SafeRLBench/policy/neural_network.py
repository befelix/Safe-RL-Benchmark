"""Neural Network Policy implementation."""

from SafeRLBench import Policy

import SafeRLBench.error as error
from SafeRLBench.error import NotSupportedException

from numpy.random import normal

try:
    import tensorflow as tf
except:
    tf = None

import logging

logger = logging.getLogger(__name__)


def default_init_weights(shape):
    """Default weights initialization."""
    weights = tf.random_normal(shape, mean=0, stddev=0.1, name='weights')
    return tf.Variable(weights)


class NeuralNetwork(Policy):
    """Fully connected Neural Network Policy.

    Attributes
    ----------
    layers : list of integers
        A list describing the layer sizes. The first element represents the
        size of the input layer, the last element the size of the output
        layer.
    state_space : space instance
    action_space : space instance
    weights : tf.Variable
        If none the init_weights function will be used to initialize the
        weights.
    init_weights : callable
        Takes a shape as an argument and returns a tf.Variable according to
        this shape.
    activation : list of activation functions
        An activation function which will be used to construct the respective
        layer. If only one activation function is passed, it will be used for
        every layer. If the argument is None by default the sigmoid function
        will be used.
    dtype : string
        Data type of input and output.
    y_pred : tensorflow op
        This is the actual neural network computing the next step.
    sess : tensorflow session
        The session the variables are initialized in. It is used to evaluate
        and update the network.
        Make sure session is set to an active session while running.
    """

    def __init__(self, layers, state_space, action_space, weights=None,
                 init_weights=None, activation=None, dtype='float',
                 scope='global', do_setup=False):
        """Initialize Neural Network wrapper."""
        if tf is None:
            raise NotSupportedException(error.NO_TF_SUPPORT)

        if (len(layers) < 2):
            raise ValueError('At least two layers needed.')

        # store arguments convenient for copy operation
        self.args = [layers, state_space, action_space]
        self.kwargs = {
            'weights': weights,
            'init_weights': init_weights,
            'activation': activation,
            'dtype': dtype
        }

        self.dtype = dtype
        self.layers = layers

        if init_weights is None:
            self.init_weights = default_init_weights
        else:
            self.init_weights = init_weights

        # Activation function
        if activation is None:
            activation = (len(layers) - 2) * [tf.sigmoid]
        elif (isinstance(activation, list)
                and (len(activation) != len(layers) - 2)):
            raise ValueError('Activation list has wrong size.')
        else:
            activation = (len(layers) - 2) * [activation]

        self.activation = activation

        # Symbols
        self.X = tf.placeholder(dtype, shape=[None, layers[0]], name='X')
        self.a = tf.placeholder(dtype, shape=[None, layers[-1]], name='a')

        if do_setup:
            with tf.variable_scope(self.scope):
                self.setup()
        else:
            # Make sure all fields exist
            self.W_action = None
            self.W_var = None
            self.a_pred = None
            self.var = None
            self.h = None

            self.is_set_up = False

    def setup(self):
        """Setup the network graph.

        The weights and graph will be initialized by this function. If do_setup
        is True, setup will automatically be called, when instantiating the
        class.
        """
        layers = self.layers
        weights = self.kwargs['weights']

        with tf.variable_scope('policy_estimator'):
            # Weights for the action estimation
            with tf.variable_scope('action_estimator'):
                if weights is None:
                    w = []
                    for i in range(len(layers) - 1):
                        w.append(self.init_weights((layers[i], layers[i + 1])))
                else:
                    w = weights

                self.W_action = w

            # Weights for variance estimation
            with tf.variable_scope('variance_estimator'):
                self.W_var = []
                for i in range(len(layers) - 1):
                    self.W_var.append(self.init_weights((layers[i], 1)))

            # generate nn tensor
            self.a_pred = self._generate_network()
            self.var = self._generate_variance()

        self.is_set_up = True

    def _generate_network(self):
        self.h = [self.X]
        for i, act in enumerate(self.activation):
            h_i = self.h[i]
            w_i = self.W_action[i]
            self.h.append(act(tf.matmul(h_i, w_i)))

        return tf.matmul(self.h[-1], self.W_action[-1])

    def _generate_variance(self):
        var = []
        for h_i, w_i in zip(self.W_var, self.h):
            var.append(tf.reduce_sum(tf.multiply(h_i, w_i)))
        return tf.abs(tf.add_n(var, name='variance'))

    def copy(self, scope, do_setup=True):
        """Generate a copy of the network for workers."""
        self.kwargs['scope'] = scope
        self.kwargs['do_setup'] = do_setup
        return NeuralNetwork(*self.args, **self.kwargs)

    def map(self, state):
        """Compute output in session.

        Make sure a default session is set when calling.
        """
        sess = tf.get_default_session()
        mean, var = sess.run([self.a_pred, self.var], {self.X: [state]})
        return normal(mean, var)

    @property
    def parameters(self):
        """Return weights of the neural network.

        This returns a list of tf.Variables. Please note that these can not
        simply be updated by assignment. See the parameters.setter docstring
        for more information.
        The list of tf.Variables can be directly accessed through the
        attribute `W`.
        """
        return self.W_action.eval()

    @parameters.setter
    def parameters(self, update):
        """Setter function for parameters.

        Since the parameters are a list of tf.Variables, instead of directly
        assigning to them you will need to pass an update tensor which updates
        the values. To create such a tensor access the `W` attribute which
        contains the weight variables and use it to instantiate an update
        tensor.
        This method will then run the update tensor in the session.
        """
        self.sess.run(*update)

    @property
    def parameter_space(self):
        """Return parameter space."""
        pass
