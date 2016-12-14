from SafeRLBench import Monitor

import logging
import sys


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
        self.n_jobs = 1
        self.monitor_verbosity = 0

    @property
    def monitor(self):
        """Lazily generate monitor as configured."""
        if not hasattr(self, '_monitor'):
            global config
            self._monitor = Monitor(self.monitor_verbosity)
        return self._monitor

    def setMonitorVerbosity(self, verbose):
        if hasattr(self, '_monitor'):
            self._monitor.verbose = verbose
        self.monitor_verbosity = verbose

    def setJobs(self, n_jobs):
        self.n_jobs = n_jobs

    def setLoggerLevel(self, level=logging.DEBUG):
        self.log.setLevel(level)

    def setLoggerStdOut(self):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s'
                                      + ' - %(message)s')
        ch.setFormatter(formatter)
        self.log.addHandler(ch)
