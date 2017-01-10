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

        self._stream_handler = None
        self._file_handler = None
        self._fmt = ('%(process)d - %(asctime)s - %(name)s - %(levelname)s'
                     + ' - %(message)s')
        self._formatter = logging.Formatter(self._fmt)

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

    def logger_set_level(self, level=logging.INFO):
        """
        Set the logger level package wide.

        Parameters
        ----------
        level :
            Logger level as defined in logging.
        """
        self.log.setLevel(level)

    @property
    def logger_stream_handler(self):
        return self._stream_handler

    @logger_stream_handler.setter
    def stream_handler(self, ch):
        if self._stream_handler is not None:
            self.log.removeHandler(self._stream_handler)

        self._stream_handler = ch
        if ch is not None:
            self.log.addHandler(ch)

    @property
    def logger_file_handler(self):
        return self._file_handler

    @logger_file_handler.setter
    def logger_file_handler(self, fh):
        if self._file_handler is not None:
            self.log.removeHandler(self._file_handler)

        self._file_handler = fh
        if fh is not None:
            self.log.addHandler(fh)

    @property
    def logger_format(self):
        return self._fmt

    @logger_format.setter
    def logger_format(self, fmt):
        self._formatter = logging.Formatter(fmt)

        if self.logger_stream_handler is not None:
            self.logger_stream_handler.setFormatter(self._formatter)

        if self.logger_file_handler is not None:
            self.logger_file_handler.setFormatter(self._formatter)

    def logger_add_stream_handler(self):
        """Set a handler to print logs to stdout."""
        if self.stream_handler is not None:
            self.log.removeHandler(self.stream_handler)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(self._formatter)

        self.stream_handler = ch
        self.log.addHandler(ch)

    def logger_add_file_handler(self, path):
        """
        Set a handler to print to file.

        Parameters
        ----------
        path :
            Path to log file.
        """
        if self.file_handler is not None:
            self.log.removeHandler(self.file_handler)

        fh = logging.FileHandler(path)
        fh.setFormatter(self._formatter)

        self.file_handler = fh
        self.log.addHandler(fh)
