Safe Reinforcement Learning Library for Python
==============================================

.. image:: https://travis-ci.com/befelix/Safe-RL-Benchmark.svg?token=gAjgFLh7fGz27Y8XYV1g&branch=master
  :target: https://travis-ci.com/befelix/Safe-RL-Benchmark
  :alt: Build Status

SafeRLBench provides an interface for algorithms, environments and policies, in
order to support a reusable benchmark environment.


Structure
---------

The main module of the library contains the base classes providing the
interface, as well as the benchmark facilities used to run and compare the
algorithms. Further it contains the three submodules that contain content that
has been implemented using the respective baseclasses.

Algorithm module ``algo``
  Contains algorithm implementations like ``PolicyGradient`` or ``SafeOpt``.
  Classes in this module are subclasses of the ``AlgorithmBase`` class.

Environment module ``envs``
  Contains environment implementations like ``LinearCar`` or ``Quadrocopter``
  environments. These are subclasses of the ``EnvironmentBase`` class.

Policy module ``policy``
  Contains policies. Although some policies are specific for the use with
  certain algorithms, they are still seperated in an individual module,
  providing a interface as definde through the ``Policy`` baseclass, since in
  there are cases in which they can be optimized through different algorithms.

Installation
------------

Dependencies
~~~~~~~~~~~~

SafeRLBench requires:

  - NumPy >= 1.7
  - SciPy >= 0.19.0
  - six >= 1.10
  - futures >= 3.0.5


Cloning
~~~~~~~

The best way to install and use this library is to clone it from the repository.

  ``git clone https://github.com/befelix/Safe-RL-Benchmark.git``

In order to use the content that has already been implemented as is, navigate
to the root directory and execute:

  ``python setup.py install``

In many cases it makes sense to extend or adapt the content. Then the develop
setup is your friend. Again, navigate to the root directory of the repository
and execute:

 ``python setup.py develop``

Getting Started
---------------

The following instructions can be executed in many ways. You may use your
favorite interactive interpreter, include it in scripts or use some form of
notebook to get started.

Optimizing a Policy
~~~~~~~~~~~~~~~~~~~

To get started let us try to optimize a policy on a very simple environment.
To accomplish this we need to make a few decisions. First we need some task that
we can solve. This is implemented in the form of environments in the ``envs``
module.

  >>> # import the linear car class
  >>> from SafeRLBench.envs import LinearCar
  >>> # get an instance with the default arguments
  >>> linear_car = LinearCar()

Ok, so far so good. Next we need a policy. Again, before anything gets too
complicated, let us take linear mapping. Fortunately there is a linear mapping
implemented in the ``policy`` module.

  >>> # import the linear policy class
  >>> from SafeRLBench.policy import LinearPolicy
  >>> # instantiate it with d_state=2 and d_action=1
  >>> policy = LinearPolicy(2, 1)

Notice that we did not use the default parameters this time. The LinearPolicy
is a linear mapping from an element of a d_state-dimensional element to a
d_action-dimensional element. Our ``linear_car`` instance with the default
arguments is just a car with a (position, velocity)-state on a line, thus our
state space is two dimensional and we can accelerate along the line, so our
action space is one dimensional.

Now we need our third and last ingredient, which is the algorithm, that optimizes
the policy on the environment. Well, an algorithm that is very stable on this
environment is the ``PolicyGradient`` algorithm with a central differences
gradient estimator.

  >>> # import the policy gradient class
  >>> from SafeRLBench.algo import PolicyGradient
  >>> # instantiate it with the environment and algorithm
  >>> optimizer = PolicyGradient(linear_car, policy, estimator='central_fd')

To make this reproducible we may want to set the initial parameters used for
the optimization. PolicyGradient will randomly initialize, if no parameters are
set, but if we just preset the parameters, it will take what it gets.

  >>> # set policy parameters
  >>> policy.parameters = [1, 1, 1]
  >>> # and optimize it
  >>> optimizer.optimizer()
