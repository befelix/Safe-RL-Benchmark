from __future__ import absolute_import

import logging

from .monitor import Monitor
from .configuration import SRBConfig

# Initialize configuration
config = SRBConfig(logging.getLogger(__name__))

from .base import EnvironmentBase, Space, AlgorithmBase, Policy, ProbPolicy
from .measure import Measure, BestPerformance
from .bench import Bench, BenchConfig
from . import algo
from . import envs
from . import tools
from . import policy

# Add things to all
__all__ = ['EnvironmentBase',
           'Space',
           'AlgorithmBase',
           'Policy',
           'ProbPolicy',
           'Monitor',
           'SRBConfig',
           'Measure',
           'BestPerformance',
           'Bench',
           'BenchConfig',
           'envs',
           'tools',
           'algo',
           'policy']


# Import test after __all__ (no documentation)
# from numpy.testing import Tester
# test = Tester().test
