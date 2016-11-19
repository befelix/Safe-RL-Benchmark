"""R^d with any shape."""
import numpy as np
from SafeRLBench import Space


class RdSpace(Space):
    """R^d Vectorspace."""

    def __init__(self, shape):
        """Initialize with shape."""
        self.shape = shape

    def contains(self, x):
        """Check if element is contained."""
        return isinstance(x, np.ndarray) and x.shape == self.shape

    def element(self):
        """Return arbitrary element."""
        return np.ones(self.shape)
