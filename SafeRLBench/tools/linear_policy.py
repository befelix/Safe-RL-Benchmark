import numpy as np

class LinearPolicy(object):
    def __init__(self, action_shape, state_shape):
        self.action_shape = action_shape
        self.state_shape  = state_shape

        self.parameters = np.emptya(state_shape)
        self.bias       = 0

    # Note
    # np.empty(action_shape + state_shape)
    # np.swapaxes(np.tile(A,B.shape).reshape(B.shape+A.shape), i, j)
    def __call__(self, state):
        
