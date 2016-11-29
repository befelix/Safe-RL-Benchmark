from __future__ import absolute_import

from .monitor import Monitor
from .configuration import SRBConfig

# Initialize configuration
config = SRBConfig()

from .base import EnvironmentBase, Space, AlgorithmBase
from . import algo
from . import envs
from . import tools

# Add everython to all
__all__ = ['EnvironmentBase', 'Space', 'AlgorithmBase', 'Monitor',
           'envs', 'tools', 'algo']


# Import test after __all__ (no documentation)
from numpy.testing import Tester
test = Tester().test
