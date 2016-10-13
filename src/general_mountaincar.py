import numpy as np
from numpy import pi

import theano.tensor as T
from theano import function, grad

import matplotlib.pyplot as plt

def isContour(contour):
    if isinstance(contour, tuple):
        if isinstance(contour[0], TensorVariable) and isinstance(contour[1], TensorVariable):
            return(True)

    return(False)

class GeneralMountainCar(object):

    def __init__(self, inital_state=np.array([0,0]), domain=np.array([-1,1]), contour=None, 
                 gravitation=0.0025, max_velocity=0.07, power=0.0015, goal=0.6):
        
        # setup environment parameters
        self.min_position  = domain[0]
        self.max_position  = domain[1]
        self.goal_position = goal

        self.max_velocity  = max_velocity
        self.power         = power

        self.gravitation   = gravitation
    
        # setup contour
        if isContour(contour):
            self.x = contour[0]
            self.y = contour[1]
        else:
            self.x = T.dscalar('x')
            self.y = -T.cos(pi*self.x)

        self.hx = function([self.x], self.y)

        self.dydx_var = grad(self.y,self.x)
        self.dydx = function([self.x], self.dydx_var)

        # init state
        self.state = inital_state
        self.initial_state = inital_state 

        # setup plot fields
        self.figure = None
        self.plot   = None
        self.point  = None

    def _update(self, action):
        """
        Computes step considering the action
        ___________
        Parameters:
        action: dscalar
        """
        force = max(min(action, 1.0), -1.0)

        position = self.state[0]
        velocity = self.state[1]

        velocity += (force * self.power - self.dydx(position) * self.gravitation)
        position += velocity

        velocity = max(min(velocity, self.max_velocity), -self.max_velocity)
        position = max(min(position, self.max_position), self.min_position)

        self.state = np.array([position, velocity])

        archieved = (position >= self.goal_position)

        return self.state, self._reward(), archieved

    def _reset(self):
        self.state = self.inital_state

    def _reward(self):
        return(self._height()-1)

    def _height(self):
        return(self.hx(self.state[0]).item())

    def _position(self):
        return(self.state[0])

    def _plot(self):
        """
        Plot contour with current position marked
        Update plot figure already exists
        """
        if self.figure is None:
            plt.ion()

            figure = plt.figure()
            plot   = figure.add_subplot(111)

            # plot contour
            c = lambda t: self.hx(t).item()
            x = np.linspace(self.min_position, self.max_position, 50)
            y = list(map(c, x))
            
            plot.plot(x, y)

            # plot car
            point, = plot.plot([self._position()], [self._height()],'or')

            # draw
            figure.canvas.draw()

            # store figure references
            self.figure = figure
            self.plot   = plot
            self.point  = point
        else:
            # update car
            point = self.point

            point.set_xdata([self._position()])
            point.set_ydata([self._height()])
           
            self.figure.canvas.draw()

    def _animate(self, action, steps):
        for n in range(steps):
            self._plot()
            self._update(action)
