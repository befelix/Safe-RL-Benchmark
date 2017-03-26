"""Policy tests."""
from __future__ import division, print_function, absolute_import

from unittest2 import TestCase

from SafeRLBench.policy import NeuralNetwork
from SafeRLBench.policy import LinearPolicy, DiscreteLinearPolicy

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
        args = [[2]]
        with self.assertRaises(ValueError):
            NeuralNetwork(*args)

        # test field existence
        args = [[2, 6, 1]]

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
        self.assertEqual(nn.dtype, 'float')

        self.assertEqual(len(nn.W_action), 2)
        self.assertEqual(len(nn.W_var), 1)

        # well... because is does not work for whatever fucking reason.
        self.assertEqual(str(type(nn.a_pred)), str(tf.Tensor))
        self.assertIn(str(type(nn.var)), (str(tf.Tensor), str(tf.constant)))

        self.assertEqual(len(nn.h), 2)

    def test_mapping(self):
        """Test NeuralNetwork mapping."""
        args = [[2, 1]]

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
        args = [[2, 1]]
        kwargs = {'do_setup': True}

        nn = NeuralNetwork(*args, **kwargs)

        with tf.Session().as_default():
            nn.parameters = nn.W_action[0].assign([[2.], [1.]])
            assert((np.array([[2.], [1.]]) == nn.parameters).all())
            self.assertEqual(nn(np.array([2., 1.])), [5.])


class TestLinearPolicy(TestCase):
    """Test the Linear Policy."""

    def test_initialization(self):
        """Test LinearPolicy initialization."""
        lp = LinearPolicy(2, 1)

        self.assertEqual(lp.d_state, 2)
        self.assertEqual(lp.d_action, 1)

        self.assertEqual(lp.par_dim, 2)
        self.assertIs(lp._par_space, None)

        self.assertFalse(lp.initialized)

        self.assertIs(lp._parameters, None)
        self.assertTrue(lp.biased)
        self.assertEqual(lp._bias, 0)
        self.assertIs(lp._par, None)

        par_mock = Mock()
        par_space_mock = Mock()

        with self.assertRaises(ValueError):
            lp_mocked = LinearPolicy(2, 1, par_mock, par_space_mock)

        par_mock.shape = (2,)
        par_mock.copy.return_value = par_mock

        lp_mocked = LinearPolicy(2, 1, par_mock, par_space_mock)

        self.assertTrue(lp_mocked.initialized)
        self.assertEqual(par_mock, lp_mocked.parameters)

        self.assertEqual(par_space_mock, lp_mocked.parameter_space)

    def test_discrete_map(self):
        """Test DiscreteLinearPolicy map."""
        dp = DiscreteLinearPolicy(2, 1, biased=False)
        dp.parameters = np.array([1, 1])
        self.assertEqual(dp([1, 1]), 1)
        self.assertEqual(dp([-1, -1]), 0)

        dp2 = DiscreteLinearPolicy(2, 2, biased=False)
        dp2.parameters = np.array([1, 1, -1, -1])
        assert(all(dp2([1, 1]) == [1, 0]))
        assert(all(dp2([-1, -1]) == [0, 1]))
