# flake8: noqa

"""
quadrocopter_classes.py

Written By: Adrian Esser and David Wu

Changes:
- Aug 2015 - Vectorized quadrotor state, moved state conversions here

This file contains all classes for the quadrocopter simulation!

This class defines an object for the state of the drone.
The state contains the position, velocity, and acceleration information,
the rotation matrix (which implies pitch, roll, and yaw), and the angular
velocity information.
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from .transformations import (quaternion_from_euler, euler_from_matrix,
                              euler_matrix)


__all__ = ['State', 'Parameters', 'StateVector']


class StateVector(object):
    """Class to provide same StateVector object as used in ROS."""

    def __init__(self):
        super(StateVector, self).__init__()
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        self.euler = np.zeros(3)
        self.omega_g = np.zeros(3)
        self.omega_b = np.zeros(3)
        self.quat = np.zeros(4)
        self.quat[3] = 1

    def asarray(self):
        # this is much faster than hstack.
        return np.array(list(self))

    def __iter__(self):
        for p in self.pos:
            yield p
        for v in self.vel:
            yield v
        for a in self.acc:
            yield a
        for e in self.euler:
            yield e
        for o in self.omega_g:
            yield o
        for o in self.omega_b:
            yield o
        for q in self.quat:
            yield q


class State:

    def __init__(self):

        self.R = np.eye(3)
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        self.omega = np.zeros(3)

    @property
    def quaternion(self):
        """Rotation quaternion corresponding to R."""
        return quaternion_from_euler(*self.rpy)

    @property
    def rpy(self):
        """Roll, pitch, yaw corresponding to R."""
        return np.array(euler_from_matrix(self.R))

    @property
    def state_vector(self):
        """Return the state as a StateVector."""
        state = StateVector()
        state.pos[:] = self.pos
        state.vel[:] = self.vel
        state.acc[:] = self.acc
        state.quat[:] = self.quaternion
        state.euler[:] = self.rpy
        state.omega_b[:] = self.omega
        state.omega_g[:] = self.R.dot(self.omega)
        return state

    def rpy_to_R(self, rpy):
        return euler_matrix(*rpy)[:3, :3]


class Parameters:
    """Parameters for quadrotor the define the physics."""

    def __init__(self):

        # m, mass of vehicle (kg)
        self.m = 1.477
        # g, mass normalized gravitational force (m/s^2)
        self.g = 9.8
        # L, vehicle arm length (m)
        self.L = 0.18
        # K, motor constant, determined experimentally
        self.K = 0.26
        # Ix, inertia around the body's x-axis (kg-m^2)
        self.Ix = 0.01152
        # Iy, inertia around the body's y-axis (kg-m^2)
        self.Iy = 0.01152
        # Iz, inertia around the body's z-axis (kg-m^2)
        self.Iz = 0.0218
        # fmin, mass normalized minimum rotor force (m/s^2)
        self.fmin = 0.17
        # fmax, mass normalized maximum rotor force (m/s^2)
        self.fmax = 6.0
        # vmax, maximum quadrotor velocity (m/s)
        self.vmax = 2.0
        # eta, damping ratio
        self.eta = 0.707
        # tau_z, time constant for vertical direction
        self.tau_z = 1.0
        # tau_Iz, integral time constant for vertical direction
        self.tau_Iz = 0.05
        # tau_yaw, time constant for yaw rate
        self.tau_yaw = 0.55
        # tau_Iyaw, integral time constant for yaw rate
        self.tau_Iyaw = 0.01
        # eta_y, damping ratio
        self.eta_y = 0.707
        # tau_y, time constant for x and y direction
        self.tau_y = 1.7
        # tau_Iu, integral time constant for x and y dir.
        self.tau_Iu = 2.5
        # tau_p, time constant for roll rate
        self.tau_p = 0.18
        # tau_q, time constant for pitch rate
        self.tau_q = 0.18
        # tau_r, time constant for yaw rate
        self.tau_r = 0.1
        # tau_rp, time constant
        self.tau_rp = 0.18
        # tau_f, time constant for force integration
        self.tau_f = 0.1

        # Air drag factor in body x direction [dimensionless]
        self.CD_bx = 0.55
        # Air drag factor in body y direction [dimensionless]
        self.CD_by = 1.25
        # Air drag factor in body z direction [dimensionless]
        self.CD_bz = 0.3

        # Air drag factor in body x direction [-]
        self.CD_bx = 0.35
        # Air drag factor in body y direction [-]
        self.CD_by = 1.25
        # Air drag factor in body z direction [-]
        self.CD_bz = 0.3

        # Delay in the signal being sent from quad to computer (us)
        self.incoming_delay = 0.0
        # Delay in signal being sent from computer to quad (us)
        self.outgoing_delay = 100000.0
        # Update rate of inner loop (us)
        self.inner_loop_cycle = 8000.0
        # Update rate of outer loop (us)
        self.outer_loop_cycle = 15000.0

        # Takeoff height (m)
        self.takeoff_height = 1.0
        # Takeoff speed (m/s)
        self.takeoff_speed = 0.25
