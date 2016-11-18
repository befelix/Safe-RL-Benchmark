import numpy as np
from SafeRLBench import Space


class RdSpace(Space):
    """
    R^d Vectorspace
    """
    def __init__(self, shape):
        self.shape = shape

    def contains(self, x):
        return isinstance(x, np.ndarray) and x.shape == self.shape

    def element(self):
        return np.ones(self.shape)
