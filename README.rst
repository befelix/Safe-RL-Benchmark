Safe Reinforcement Learning Library for Python
==============================================

.. image:: https://travis-ci.com/befelix/Safe-RL-Benchmark.svg?token=gAjgFLh7fGz27Y8XYV1g&branch=master
  :target: https://travis-ci.com/befelix/Safe-RL-Benchmark
  :alt: Build Status

.. image:: https://readthedocs.org/projects/saferlbench/badge/?version=latest
  :target: http://saferlbench.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://codecov.io/gh/befelix/Safe-RL-Benchmark/coverage.svg?branch=master
  :target: https://codecov.io/gh/befelix/Safe-RL-Benchmark?branch=master
  :alt: Coverage

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

There is also a notebook in the examples directory which contains the examples
below.

Optimizing a Policy
~~~~~~~~~~~~~~~~~~~

To get started let us try to optimize a policy on a very simple environment.
To accomplish this we need to make a few decisions. First we need some task to
solve. This is implemented in the form of environments in the ``envs``
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
  >>> # setup some initial parameters
  >>> policy.parameters = [1, 1, 1]

Notice that we did not use the default parameters this time. The LinearPolicy
is a linear mapping from an element of a d_state-dimensional space to a
d_action-dimensional space. Our ``linear_car`` instance with the default
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

Earlier we setup some initial parameters. The `PolicyGradient` optimizer will
check if there are initial parameters and use those if present. If there are
no preset parameters he will randomly initialize them, until he finds a
nonzero gradient.

  >>> # optimize the policy when everything is set up.
  >>> optimizer.optimizer()

Now the algorithm might run for a while depending on how much effort the
optimization takes. Unfortunately no information on the progress shows up, yet.
We will deal with that in the next part.

Lets take a look at what actually happened during the run. For this we can
access the `monitor` and generate some plots. For example, we could plot the
reward evolution during optimization.

  >>> # use matplotlib for plotting
  >>> import matplotlib.pyplot as plt
  >>> # retrieve the rewards
  >>> y = optimizer.monitor.rewards
  >>> plt.plot(range(len(y)), y)
  >>> plt.show()

Configuration
~~~~~~~~~~~~~

Especially when you try to set up a new environment it is often very useful
to get some logging information. In `SafeRLBench` there is an easy way to
setup some global configurations. Let us access the global `config` variable:

  >>> # import the config variable
  >>> from SafeRLBench import config

Well, thats it. The `config` variable is an instance of the class `SRBConfig`,
which contains methods to manipulate the overall behaviour. For example we can
easily make the logger print to stdout:

  >>> # output to stdout
  >>> config.logger_set_stream_handler()

Or we might want to change the level of the logger:

  >>> # print debug information
  >>> config.logger_set_level(config.DEBUG)

There are some more tricks and tweaks to it, which can be found directly in the
class documentation. For example we can directly assign a handler or we can
add an additional file handler that writes our output to a file, etc. For more
information on that refer to the documentation.

In general the class methods and attributes will follow the a naming
convention, that is, the first part of the name will regard the part we want
to configure and the second part will describe what we want to change.

Apart from the logger, let's say we want to change the amount of jobs that are
used by the benchmarking facility. (We will see it in the next section.)
Simply configure it with:

  >>> # set number of jobs to 4
  >>> config.jobs_set(4)

Or set the verbosity level of the monitor:

  >>> # increase verbosity to 2
  >>> config.monitor_set_verbosity(2)

Benchmarking
~~~~~~~~~~~~

We can run optimize policies on environments now, the next thing we want to do
is benchmarking. For this we can use the benchmark facilities that the
library provides. In order to run a benchmark, we need to produce an instance
``BenchConfig``.

When we take a look at the documentation of this class, it takes two arguments.
The first one is ``algs`` the second one ``envs``. And now it gets a litte bit
weird, both of them are a list of a list of tuples where the second element is
a list of dictionaries. Confused? Yes, but here is a simple example:

>>> # define environment configuration.
>>> envs = [[(LinearCar, {'horizon': 100})]]
>>> # define algorithms configuration.
>>> algs = [[
...   (PolicyGradient, [{
...     'policy': LinearPolicy(2, 1, par=[1, 1, 1]),
...     'estimator': 'central_fd',
...     'var': var
...   } for var in [1, 1.5, 2, 2.5]])
... ]]

Ok, so here we did not really use the outer lists, but in the case of
reinforcement learning we have the problem, that many environments will require
individual configuration of an algorithm. The structure of ``envs`` and
``algs`` allows for a lot of flexibility, although in many cases the outer-most
list will not be needed.

So what happens? The outer most lists of envs and algs will get zipped, such
that we can support pair wise configurations. Further, the tuple contains a
class in the first element and a list of configurations dictionaries in the
second element. This essentially allows quick generation of many configurations
for a single algorithm or environment. Finally the cartesian product of **all**
configurations in the inner lists will be executed by the ``Bench``.

So in the example above, we only have a single environment configuration,
but the corresponding list in ``algs`` contains four configurations for the
``PolicyGradient``. Overall this will result in four test runs.

In case we had

>>> envs_two = [(LinearCar, {'horizon': 100}), (LinearCar, {'horizon': 200})]

``BenchConfig`` would supply eight configurations to the ``Bench``. By the way,
in case we do not need the outer most list, we could even leave it away.

>>> # instantiate BenchConfig
>>> config = BenchConfig(algs, envs)

Next we can evaluate the configuration achieving the best performance. Again,
the library contains a tool for this, the measures.

>>> # import the best performance measure
>>> from SafeRLBench.measures import BestPerformance
>>> # import the Bench
>>> from SafeRLBench import Bench
>>> # instantiate the bench
>>> bench = Bench(config, BestPerformance())
