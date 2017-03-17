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
    """Dependency decorator.

    Decorates a class with a dependency decorator, that will raise a
    `NotSupportedException` when `dep` is None.

    Parameters
    ----------
    dep : Module
        The dependent module.
    dep_name : String
        Name of the dependency for a meaningful error message.

    Notes
    -----
    Make sure not to call super within the class when using the decorator.
    """
    def dependency(dep_cls):
        class DependentClass(object):
            def __new__(cls, *args, **kwargs):
                if dep is None:
                    raise NotSupportedException(dep_name + ' is not installed.')
                return dep_cls(*args, **kwargs)
        DependentClass.__name__ = dep_cls.__name__
        return DependentClass
    return dependency
