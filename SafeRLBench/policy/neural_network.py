"""Neural Network Policy implementation."""

from SafeRLBench import Policy

import logging

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
except:
    logger.warning("TensorFlow is not installed.")


def init_weights(shape):
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
    weights : tf.Variable
        If none the init_weights function will be used to initialize the
        weights.
    init_weights : callable
        takes a shape as an argument and returns a tf.Variable according to
        this shape.
    activation : list of activation functions
        an activation function which will be used to construct the respective
        layer. If only one activation function is passed, it will be used for
        every layer.
    dtype : string
        data type of input and output.
    """

    def __init__(self, layers, weights=None, init_weights=init_weights,
                 activation=tf.sigmoid, dtype='float'):
        """Initialize Neural Network wrapper."""
        if (len(layers) < 2):
            raise ValueError('At least two layers needed.')

        self.layers = layers
        self.init_weights = init_weights

        # Activation function
        if (isinstance(activation, list)
                and (len(activation) != len(layers) - 1)):
            raise ValueError('Activation list has wrong size.')
        else:
            activation = (len(layers) - 1) * [activation]

        self.activation = activation

        # Symbols
        self.X = tf.placeholder(dtype, shape=[None, layers[0]], name='X')
        self.y = tf.placeholder(dtype, shape=[None, layers[-1]], name='y')

        # Weights
        if weights is None:
            w = []
            for i in range(len(layers) - 1):
                with tf.variable_scope('weights'):
                    w.append(self.init_weights((layers[i], layers[i + 1])))
        else:
            w = weights

        self.W = w

        # generate nn tensor
        self.y_pred = self._generate_network()

        # initialize tf session
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _generate_network(self):
        h = [self.X]
        for i in range(len(self.layers) - 1):
            act = self.activation[i]
            h_i = h[i]
            w_i = self.W[i]
            h.append(act(tf.matmul(h_i, w_i)))
        return h[-1]

    def map(self, state):
        """Compute output in session."""
        return self.sess.run(self.y_pred, {self.X: [state]})

    @property
    def parameters(self):
        """Return weights of the neural network.

        This returns a list of tf.Variables. Please note that these can not
        simply be updated by assignment. See the parameters.setter docstring
        for more information.
        The list of tf.Variables can be directly accessed through the
        attribute `W`.
        """
        return self.sess.run(self.W)

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
        self.sess.run(update)

    @property
    def parameter_space(self):
        """Return parameter space."""
        pass
