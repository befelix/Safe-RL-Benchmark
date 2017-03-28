"""Algorithm Module.

=================== =========================================
Algorithm
=============================================================
A3C                 Asynchronous Actor-Critic Agents
PolicyGradient      Different Policy Gradient Implementations
DiscreteQLearning   Q-Learning using a table
SafeOpt             Bayesian Optimization with SafeOpt
=================== =========================================
"""

from .policygradient import PolicyGradient
from .safeopt import SafeOpt
from .a3c import A3C
from .q_learning import DiscreteQLearning

__all__ = ['PolicyGradient', 'SafeOpt', 'A3C', 'DiscreteQLearning']
