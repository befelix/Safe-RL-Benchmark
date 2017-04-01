Environments
============

This module contains environment implementations. Each environment has to
inherit from the `EnvironmentBase` class and should be accessed through
the base class interface.

Overview
--------

=================== =================================== =======================
Environment         State Space                         Action Space
=================== =================================== =======================
GeneralMountainCar  :math:`[-1,1]\times[-0.07,0.07]`    :math:`[-1, 1]`
GymWrap
LinearCar           :math:`\mathbb{R}^{2d}`             :math:`[-1, 1]^d`
MDP
Quadrocopter
=================== =================================== =======================

Implementing an Environment
---------------------------

When implementing an environment a couple of things have to be considered.
`EnvironmentBase` is an abstract base class. It will require any subclass to
implement certain private methods which will be invoked by the public
interface. Further certain attributes should be initialized, also as specified
below, to support monitoring the execution.

Requirements
~~~~~~~~~~~~

Environments have to inherit from `SafeRLBench.EnvironmentBase`.

=============== =============== ===============================================
Initialize Attributes
===============================================================================
state_space     Space object
action_space    Space object
horizon         Integer         Used in default _rollout implementation.
=============== =============== ===============================================

=============== =============== ===============================================
Must implement
===============================================================================
_update         action          Returns (action, state, reward)
_reset
=============== =============== ===============================================

=============== =============== ===============================================
May implement
===============================================================================
_rollout        policy          Returns list of (action, state, reward)
=============== =============== ===============================================
