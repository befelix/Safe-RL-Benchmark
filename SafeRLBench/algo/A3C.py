"""Asynchronous Actor-Critic Agents."""

from SafeRLBench import AlgorithmBase

import logging

logger = logging.getLogger(__name__)


class A3C(AlgorithmBase):
    """Implementation of the Asynchronous Actor-Critic Agents Algorithm.

    Attributes
    ----------
    """

    def __init__(self, environment, policy, max_it, num_worker=2):
        """Initialize A3C."""
        if not hasattr(policy, 'sess'):
            raise ValueError('Policy needs `sess` attribute.')

        super(A3C, self).__init__(environment, policy, max_it)

        # init value function
        # init advantage op
        # init loss op

    def _initialize(self):
        pass

    def _step(self):
        pass

    def _is_finished(self):
        pass

    def _optimize(self):
        pass
