"""Policy tests."""
from __future__ import division, print_function, absolute_import

from unittest2 import TestCase

from SafeRLBench.policy import NeuralNetwork

import numpy as np
import tensorflow as tf

from mock import Mock
import logging

logger = logging.getLogger(__name__)


class TestNeuralNetwork(TestCase):
    """Test the Neural Netork Policy."""

    def test_initialization(self):
        """Test NeuralNetwork initialization."""
        # test bad layer size:
        args = [[2], Mock(), Mock()]
        with self.assertRaises(ValueError):
            NeuralNetwork(*args)

        # test field existence
        args = [[2, 6, 1], Mock(), Mock()]

        fields = ['args', 'kwargs', 'action_space', 'state_space', 'dtype',
                  'layers', 'scope', 'init_weights', 'activation', 'X', 'a',
                  'W_action', 'W_var', 'a_pred', 'var', 'h', 'is_set_up']
        nn = NeuralNetwork(*args)

        for field in fields:
            assert hasattr(nn, field)

        # test network setup
        kwargs = {
            'do_setup': True
        }

        nn = NeuralNetwork(*args, **kwargs)

        # check field contents.
        assert(all([a == b for a, b in zip(args, nn.args)]))
        self.assertEqual(nn.layers, args[0])
        self.assertEqual(nn.state_space, args[1])
        self.assertEqual(nn.action_space, args[2])
        self.assertEqual(nn.dtype, 'float')

        self.assertEqual(len(nn.W_action), 2)
        self.assertEqual(len(nn.W_var), 1)

        # well... when is does not work for whatever fucking reason.
        self.assertEqual(str(type(nn.a_pred)), str(tf.Tensor))
        self.assertIn(str(type(nn.var)), (str(tf.Tensor), str(tf.constant)))

        self.assertEqual(len(nn.h), 2)

    def test_mapping(self):
        """Test NeuralNetwork mapping."""
        action_space = Mock()
        action_space.shape = (1,)
        args = [[2, 1], Mock(), action_space]

        kwargs = {
            'weights': [tf.constant([2., 1.], shape=(2, 1))],
            'do_setup': True,
        }

        nn = NeuralNetwork(*args, **kwargs)

        sess = tf.Session()

        with sess.as_default():
            self.assertEqual(nn(np.array([2., 1.])), [5.])

    def test_variable_assignment(self):
        """Test NeuralNetwork parameter assignment."""
        action_space = Mock()
        action_space.shape = (1,)

        args = [[2, 1], Mock(), action_space]
        kwargs = {'do_setup': True}

        nn = NeuralNetwork(*args, **kwargs)

        with tf.Session().as_default():
            nn.parameters = nn.W_action[0].assign([[2.], [1.]])
            assert((np.array([[2.], [1.]]) == nn.parameters).all())
            self.assertEqual(nn(np.array([2., 1.])), [5.])
