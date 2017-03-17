"""Exceptions and error messages."""

import logging

logger = logging.getLogger(__name__)


NO_TF_SUPPORT = "TensorFlow is not installed."
NO_SO_SUPPORT = "SafeOpt is not installed."
NO_GPy_SUPPORT = "GPy is not installed."


class NotSupportedException(Exception):
    """Exception raised when requirements are not installed."""

    pass


class MultipleCallsException(Exception):
    """Exception raised when a setup method is called multiple times."""

    pass


def add_dependency(dep, dep_name='Some'):
    """Dependency decorator."""
    def dependency(cls):
        class DependentClass(object):
            def __init__(self, *args, **kwargs):
                if dep is None:
                    raise NotSupportedException(dep_name + 'is not installed.')
                return cls(*args, **kwargs)
        return DependentClass
    return dependency
