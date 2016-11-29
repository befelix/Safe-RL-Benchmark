from SafeRLBench import Monitor


class SRBConfig(object):
    def __init__(self):
        """Initialize default configuration."""
        pass

    @property
    def monitor(self):
        if not hasattr(self, '_monitor'):
            global config
            self._monitor = Monitor()
        return self._monitor
