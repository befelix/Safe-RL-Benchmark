from __future__ import absolute_import

import logging

from .configuration import SRBConfig

# Initialize configuration
config = SRBConfig(logging.getLogger(__name__))

from .monitor import AlgoMonitor, EnvMonitor
from .base import EnvironmentBase, Space, AlgorithmBase, Policy, ProbPolicy
from .measure import Measure, BestPerformance
from .bench import Bench, BenchConfig
from . import algo
from . import envs
from . import policy
from . import error

# Add things to all
__all__ = ['EnvironmentBase',
           'Space',
           'AlgorithmBase',
           'Policy',
           'ProbPolicy',
           'AlgoMonitor',
           'EnvMonitor',
           'SRBConfig',
           'Measure',
           'BestPerformance',
           'Bench',
           'BenchConfig',
           'envs',
           'algo',
           'policy',
           'error']


# Import test after __all__ (no documentation)
# from numpy.testing import Tester
# test = Tester().test
