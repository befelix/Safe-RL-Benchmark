import numpy as np
from SafeRLBench import Space


class BoundedSpace(Space):
    def __init__(self, lower, upper):
        if (np.isscalar(lower) and np.isscalar(upper)):
            self.lower = np.array([lower])
            self.upper = np.array([upper])
        else:
            assert(lower.shape == upper.shape)
            self.lower = lower
            self.upper = upper

    def contains(self, x):
        return (x.shape == self.lower.shape
                and (x > self.lower).all()
                and (x < self.lower).all())

    def element(self):
        return self.upper
