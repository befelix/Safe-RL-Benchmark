"""Import guards and exceptions."""

import logging

logger = logging.getLogger(__name__)

__all__ = ('import_failed')


failed = []


def import_failed(name):
    """Will register a failed import and log one warning."""
    global failed
    if name not in failed:
        failed.append(name)
        logger.warning(name + " is not installed.")
