"""Bounded subspace of R^n."""
import numpy as np
from SafeRLBench import Space

from numpy.random import rand


class BoundedSpace(Space):
    """
    Bounded subspace of R^n.

    Usage:
    BoundedSpace(np.array(-1,-2), np.array(1,0))
    or
    BoundedSpace(-1, 1)
    """

    def __init__(self, lower, upper, shape=None):
        """Initialize BoundedSpace."""
        if (np.isscalar(lower) and np.isscalar(upper)):
            self.lower = np.zeros(shape) + lower
            self.upper = np.zeros(shape) + upper
        else:
            self.lower = lower
            self.upper = upper

        self._dim = None

    def contains(self, x):
        """Check if element is contained."""
        return (x.shape == self.lower.shape
                and (x >= self.lower).all()
                and (x <= self.upper).all())

    def element(self):
        """Return element."""
        element = rand(*self.shape) * (self.upper - self.lower) + self.lower
        return element

    @property
    def shape(self):
        """Return element shape."""
        return self.lower.shape

    @property
    def dimension(self):
        """Return dimension of the space."""
        if self._dim is None:
            d = 1
            for i in range(len(self.shape)):
                d *= self.shape[i]
            self._dim = d
        return self._dim

    def __repr__(self):
        return 'BoundedSpace(lower=%s, upper=%s)' % (str(self.lower),
                                                     str(self.upper))
