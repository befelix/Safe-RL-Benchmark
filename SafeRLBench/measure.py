"""
Define Measurements.
"""

from abc import ABCMeta, abstractmethod
from six import add_metaclass

from operator import itemgetter


@add_metaclass(ABCMeta)
class Measure(object):
    """
    Abstract Base class defining the interface for any measurement.

    Any implementation should implement `eval` and store the result in out.

    Methods
    -------

    eval(runs)
        Evaluate a list of runs and store result in out

    Attributes
    ----------

    out : any
        Output produced by eval
    """

    def __init__(self):
        self.out = None

    @abstractmethod
    def eval(self, runs):
        """
        Evaluate a list of runs.

        Parameters
        ----------

        runs : List of BenchRun instances
            May be any subset of BenchRun instances passed in a list
        """
        pass


class BestPerformance(Measure):
    """Find the best performance achieved within runs."""

    def eval(self, runs):
        print(runs)
        # create a list of tuples with the max reward for each run
        runs_tup = []
        for run in runs:
            monitor = run.get_alg_monitor()
            max_reward = max(monitor.rewards)
            runs_tup.append((run, max_reward))

        # sort list
        sorted(runs_tup, key=itemgetter(1), reverse=False)

        self.out = runs_tup
