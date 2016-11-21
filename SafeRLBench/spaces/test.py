"""Tests for spaces module."""
from __future__ import absolute_import

from functools import partial
import inspect

from numpy import array
import SafeRLBench.spaces as spaces


"""Dictionary storing initialization arguments for classes."""
class_arguments = {
    spaces.BoundedSpace: [array([-1, -2]), array([1, 0])],
    spaces.RdSpace: [(3, 2)]
}


def add_tests():
    """Generate tests for spaces implementations."""
    for name, c in inspect.getmembers(spaces):
        if inspect.isclass(c):
            check = partial(check_contains)
            check.description = "Test implmemetation of " + c.__name__
            yield check, c


def check_contains(c):
    """Check if contains and element is implemented."""
    space = c(*class_arguments[c])
    try:
        x = space.element()
        b = space.contains(x)
    except NotImplementedError:
        assert(False)
    assert(b)
