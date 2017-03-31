"""Global Configuration Class."""
import logging
import sys


# TODO: Config: documentation, missing tests
class SRBConfig(object):
    """SafeRLBench configuration class.

    Attributes
    ----------
    logger_stream_handler :
        This is a property wrapping the current stream handler. The current
        stream handler can be accessed through this property, or it may even
        be replaced with a new stream handler. In case of resetting the stream
        handler, the old handler will be removed from the logger
        automatically.
    logger_file_handler :
        This is a property wrapping the current file handler. The current
        file handler can be accessed through this property, or it may even
        be replaced with a new stream handler. In case of resetting the file
        handler, the old handler will be removed from the logger
        automatically.
    logger_format :
        This is a property to access the format stored. This is the default
        format that will be used when adding the default handlers.
        When assigned to, the formats of already set loggers will be changed
        to the new format.
    log :
        The logger object.
    n_jobs :
        Number of jobs used by the library
    monitor_verbosity :
        Verbosity of the monitor.

    Methods
    -------
    monitor_set_verbosity :
    jobs_set :
    logger_set_level :
    logger_add_stream_handler :
    logger_add_file_handler :

    Notes
    -----
    Access logger levels through the static variables:

    ----------- -----------------
    DEBUG       logging.DEBUG
    INFO        logging.INFO
    WARNING     logging.WARNING
    ERROR       logging.ERROR
    CRITICAL    logging.CRITICAL
    ----------- -----------------
    """

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

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

        self._id_cnt = 0

        try:
            import gym
            gym.undo_logger_setup()
        except:
            pass

    def monitor_set_verbosity(self, verbose):
        """Set monitor verbosity level.

        Parameters
        ----------
        verbose : int
            Non negative verbosity level
        """
        if verbose < 0:
            raise ValueError('Verbosity level can not be negative.')
        self.monitor_verbosity = verbose

    def jobs_set(self, n_jobs):
        """Set the amount of jobs used by a worker pool.

        Parameters
        ----------
        n_jobs : Int
            Number of jobs, needs to be larger than 0.
        """
        if n_jobs <= 0:
            raise ValueError('Number of jobs needs to be larger than 0.')
        self.n_jobs = n_jobs

    def logger_set_level(self, level=logging.INFO):
        """Set the logger level package wide.

        Parameters
        ----------
        level :
            Logger level as defined in logging.
        """
        self.log.setLevel(level)

    @property
    def logger_stream_handler(self):
        """Property storing the current stream handler.

        If overwritten with a new stream handler, the logger will be updated
        with the new stream handler.

        Examples
        --------
        ch = logging.StreamHandler(sys.stdout)
        config.logger_stream_handler = ch
        """
        return self._stream_handler

    @logger_stream_handler.setter
    def logger_stream_handler(self, ch):
        """Setter method for logger_stream_handler propert."""
        if self._stream_handler is not None:
            self.log.removeHandler(self._stream_handler)

        self._stream_handler = ch
        if ch is not None:
            self.log.addHandler(ch)

    @property
    def logger_file_handler(self):
        """Property storing the current file handler.

        If overwritten with a new file handler, the logger will be updated with
        the new file handler.
        """
        return self._file_handler

    @logger_file_handler.setter
    def logger_file_handler(self, fh):
        """Setter method for logger_file_handler property."""
        if self._file_handler is not None:
            self.log.removeHandler(self._file_handler)

        self._file_handler = fh
        if fh is not None:
            self.log.addHandler(fh)

    @property
    def logger_format(self):
        """Property for default logger format.

        If overwritten stream and file handler will be updated accordingly.
        However if manually updating stream and file handler logger_format will
        be ignored.
        """
        return self._fmt

    @logger_format.setter
    def logger_format(self, fmt):
        """Setter method for logger_format property."""
        self._formatter = logging.Formatter(fmt)

        if self.logger_stream_handler is not None:
            self.logger_stream_handler.setFormatter(self._formatter)

        if self.logger_file_handler is not None:
            self.logger_file_handler.setFormatter(self._formatter)

    def logger_add_stream_handler(self):
        """Set a handler to print logs to stdout."""
        if self._stream_handler is not None:
            self.log.removeHandler(self._stream_handler)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(self._formatter)

        self._stream_handler = ch
        self.log.addHandler(ch)

    def logger_add_file_handler(self, path):
        """Set a handler to print to file.

        Parameters
        ----------
        path :
            Path to log file.
        """
        if self._file_handler is not None:
            self.log.removeHandler(self.file_handler)

        fh = logging.FileHandler(path)
        fh.setFormatter(self._formatter)

        self._file_handler = fh
        self.log.addHandler(fh)
