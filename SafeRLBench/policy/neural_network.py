"""Neural Network Policy implementation."""

from SafeRLBench import Policy

import logging

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
except:
    logger.warning("TensorFlow is not installed.")


def init_weights(shape):
    weights = tf.random_normal(shape, mean=0, stddev=0.1)
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
            self.W = []
            for i in range(len(layers) - 1):
                self.W.append(self.init_weights((layers[i], layers[i + 1])))
        else:
            self.W = weights

        # generate nn tensor
        self.y_pred = self.generate_network()

        # initialize tf session
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def generate_network(self):
        h = [self.X]
        for i in range(len(self.layers) - 1):
            act = self.activation[i]
            h_i = h[i]
            w_i = self.W[i]
            h.append(act(tf.matmul(h_i, w_i)))
        return h[-1]

    def map(self, state):
        return self.sess.run(self.y_pred, {self.X: [state]})

    @property
    def parameters(self):
        pass

    @parameters.setter
    def parameters(self, par):
        pass

    @property
    def parameter_space(self):
        pass
