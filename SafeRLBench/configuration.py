from SafeRLBench import Monitor


class SRBConfig(object):
    """
    SafeRLBench configuration class.

    Attributes
    ----------
    monitor :
        Lazily generate monitor as configured.
    """
    def __init__(self):
        """Initialize default configuration."""
        pass

    @property
    def monitor(self):
        """Lazily generate monitor as configured."""
        if not hasattr(self, '_monitor'):
            global config
            self._monitor = Monitor()
        return self._monitor
