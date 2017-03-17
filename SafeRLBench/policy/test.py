"""Policy tests."""
from __future__ import division, print_function, absolute_import

from unittest2 import TestCase

from SafeRLBench.policy import NeuralNetwork

from mock import Mock
import logging

logger = logging.getLogger(__name__)


class TestNeuralNetwork(TestCase):
    """Test the Neural Netork Policy."""

    def test_initialization(self):
        """Test neural network initialization."""
        # test bad layer size:
        args = [[2], Mock(), Mock()]
        with self.assertRaises(ValueError):
            NeuralNetwork(*args)

        # test field existence
        args = [[2, 6, 1], Mock(), Mock()]

        fields = ['args', 'kwargs', 'action_space', 'state_space', 'dtype',
                  'layers', 'init_weights', 'activation', 'X', 'a',
                  'W_action', 'W_var', 'a_pred', 'var', 'h', 'is_set_up',
                  'sess']
        nn = NeuralNetwork(*args)

        for field in fields:
            assert hasattr(nn, field)
