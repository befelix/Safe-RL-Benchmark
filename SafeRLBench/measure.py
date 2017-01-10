"""Define Measurements."""

from abc import ABCMeta, abstractmethod
from six import add_metaclass

from operator import itemgetter


@add_metaclass(ABCMeta)
class Measure(object):
    """
    Abstract Base class defining the interface for any measurement.

    The methods below are abstract and need to be implemented by any child.

    Methods
    -------

    __call__(runs)
        Abstract! Evaluate a list of runs.
    result()
        Abstract! Return the result of the evaluation.
    """

    @abstractmethod
    def __call__(self, runs):
        """
        Evaluate a list of runs.

        Parameters
        ----------

        runs : List of BenchRun instances
            May be any subset of BenchRun instances passed in a list
        """
        pass

    @abstractmethod
    def result(self):
        """Return the result of evaluation."""
        pass


class BestPerformance(Measure):
    """Find the best performance achieved within runs."""

    def __call__(self, runs):
        """
        Sort content of runs by performance.

        This class creates a tuple with a BenchRun and its respective best
        performance and then stores in a descending sorted list.
        The results are accessible through the result method.

        Parameters
        ----------

        runs : List of BenchRun instances
            May be any subset of BenchRun instances in a list.
        """
        # create a list of tuples with the max reward for each run
        runs_tup = []
        for run in runs:
            monitor = run.get_alg_monitor()
            max_reward = max(monitor.rewards)
            runs_tup.append((run, max_reward))

        # sort list
        self._result = sorted(runs_tup, key=itemgetter(1), reverse=True)

    @property
    def result(self):
        """Property to store result."""
        if not hasattr(self, '_result'):
            self._result = None
        return self._result
