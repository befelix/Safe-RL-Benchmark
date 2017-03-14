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
