"""General Mountain Car."""
from __future__ import division, absolute_import

import numpy as np
from numpy import pi, array, copy

import theano.tensor as T
from theano.tensor import TensorVariable
from theano import function, grad

import matplotlib.pyplot as plt

from SafeRLBench import EnvironmentBase
from SafeRLBench.spaces import BoundedSpace, RdSpace


def isContour(contour):
    """Check if contour is a valid contour."""
    if isinstance(contour, tuple):
        if (isinstance(contour[0], TensorVariable)
                and isinstance(contour[1], TensorVariable)):
            return(True)

    return(False)


class GeneralMountainCar(EnvironmentBase):
    """Implementation of a GeneralMountainCar Environment."""

    def __init__(self,
                 state_space=BoundedSpace(array([-1, -0.07]),
                                          array([1, 0.07])),
                 action_space=RdSpace((1,)), state=np.array([0, 0]),
                 contour=None, gravitation=0.0025, power=0.0015,
                 goal=0.6, horizon=100):
        """
        Initialize EnvironmentBase parameters and other additional parameters.

        Baseclass Parameters as in base.py.

        Parameters:
        -----------
        state: array-like with shape (2,)
            Initial state
        contour: tuple of TensorVariables
            If contour is None, a default shape will be generated.
            A valid needs to contain a dscalar as the first element
            and some function depending on the first element in the
            second element of the tuple.
        gravitation: double
        power: double
        goal: double
            Goal along x-coordinate
        """
        # Initialize Environment Base Parameters
        self.state_space = state_space
        self.action_space = action_space
        self.horizon = horizon

        # setup environment parameters
        self.goal = goal
        self.power = power
        self.gravitation = gravitation

        # setup contour
        if isContour(contour):
            self.x = contour[0]
            self.y = contour[1]
        else:
            self.x = T.dscalar('x')
            self.y = -T.cos(pi * self.x)

        self.hx = function([self.x], self.y)

        self.dydx_var = grad(self.y, self.x)
        self.dydx = function([self.x], self.dydx_var)

        # init state
        self.state = copy(state)
        self.initial_state = state

        # setup plot fields
        self.figure = None
        self.plot = None
        self.point = None

    def _update(self, action):
        """Compute step considering the action."""
        action = max(min(action, 1.0), -1.0)

        position = self.state[0]
        velocity = self.state[1]

        velocity += (action * self.power
                     - self.dydx(position) * self.gravitation)
        position += velocity

        bounds = self.state_space

        velocity = max(min(velocity, bounds.upper[1]), bounds.lower[1])
        position = max(min(position, bounds.upper[0]), bounds.lower[0])

        self.state = np.array([position, velocity])

        return action, copy(self.state), self._reward()

    def _reset(self):
        self.state = copy(self.initial_state)

    def _reward(self):
        return(self.height() - 1)

    def _rollout(self, policy):
        self.reset()
        trace = []
        for n in range(self.horizon):
            action = policy(self.state)
            trace.append(self.update(action))
            if (self.position() >= self.goal):
                return trace
        return trace

    def height(self):
        """Compute current height."""
        return(self.hx(self.state[0].item()).item())

    def position(self):
        """Compute current position in x."""
        return(self.state[0])

    # Everything below will either be removed or completly changed
    def plot(self):
        """
        Plot contour with current position marked.

        Update plot figure if already exists.
        """
        if self.figure is None:
            plt.ion()

            figure = plt.figure()
            plot = figure.add_subplot(111)

            # plot contour
            def draw_contour(t):
                return self.hx(t).item()
            x = np.linspace(self.min_position, self.max_position, 50)
            y = list(map(draw_contour, x))

            plot.plot(x, y)

            # plot car
            point, = plot.plot([self._position()], [self._height()], 'or')

            # draw
            figure.canvas.draw()

            # store figure references
            self.figure = figure
            self.plot = plot
            self.point = point
        else:
            # update car
            point = self.point

            point.set_xdata([self._position()])
            point.set_ydata([self._height()])

            self.figure.canvas.draw()

    def animate(self, action, steps):
        """To be deleted."""
        for n in range(steps):
            self._plot()
            self._update(action)
