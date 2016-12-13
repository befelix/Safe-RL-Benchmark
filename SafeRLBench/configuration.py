from SafeRLBench import Monitor

import logging
import sys

logger = logging.getLogger(__name__)


class SRBConfig(object):
    """
    SafeRLBench configuration class.

    Attributes
    ----------
    monitor :
        Lazily generate monitor as configured.
    """
    def __init__(self, log):
        """Initialize default configuration."""
        self.log = log

    @property
    def monitor(self):
        """Lazily generate monitor as configured."""
        if not hasattr(self, '_monitor'):
            global config
            self._monitor = Monitor()
        return self._monitor

    def setLoggerLevel(self, level=logging.DEBUG):
        self.log.setLevel(level)

    def setLoggerStdOut(self):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s'
                                      + ' - %(message)s')
        ch.setFormatter(formatter)
        self.log.addHandler(ch)
