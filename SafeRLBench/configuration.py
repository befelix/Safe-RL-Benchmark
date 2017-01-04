"""Global Configuration Class."""
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

    def monitor_set_verbosity(self, verbose):
        """
        Set monitor verbosity level.

        Parameters
        ----------
        verbose : int
            Non negative verbosity level
        """
        if verbose < 0:
            raise ValueError('Verbosity level can not be negative.')
        if hasattr(self, '_monitor'):
            self._monitor.verbose = verbose
        self.monitor_verbosity = verbose

    def jobs_set(self, n_jobs):
        """
        Set the amount of jobs used by a worker pool.

        Parameters
        ----------
        n_jobs : Int
            Number of jobs, needs to be larger than 0.
        """
        if n_jobs <= 0:
            raise ValueError('Number of jobs needs to be larger than 0.')
        self.n_jobs = n_jobs

    def logger_set_level(self, level=logging.DEBUG):
        """
        Set the logger level package wide.

        Parameters
        ----------
        level :
            Logger level as defined in logging.
        """
        self.log.setLevel(level)

    def logger_add_stream_handler(self):
        """Set a handler to print logs to stdout."""
        ch = logging.StreamHandler(sys.stdout)
        fmt = '%(process)d - %(asctime)s - %(name)s - %(levelname)s'
        formatter = logging.Formatter(fmt + ' - %(message)s')
        ch.setFormatter(formatter)
        self.log.addHandler(ch)

    def logger_add_file_handler(self, path):
        """
        Set a handler to print to file.

        Parameters
        ----------
        path :
            Path to log file.
        """
        fh = logging.FileHandler(path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s'
                                      + ' - %(message)s')
        fh.setFormatter(formatter)
        self.log.addHandler(fh)
