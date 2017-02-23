"""Algorithm Tests."""

from SafeRLBench.algo import PolicyGradient
from .policygradient import CentralFDEstimator, estimators

from unittest2 import TestCase
from mock import MagicMock, Mock


class TestPolicyGradient(TestCase):
    """PolicyGradientTestClass."""

    def test_pg_init(self):
        """Test initialization."""
        env_mock = MagicMock()
        pol_mock = Mock()

        for key, item in estimators.items():
            pg = PolicyGradient(env_mock, pol_mock, estimator=key)
            self.assertIsInstance(pg.estimator, item)

        pg = PolicyGradient(env_mock, pol_mock, estimator=CentralFDEstimator)
        self.assertIsInstance(pg.estimator, CentralFDEstimator)

        self.assertRaises(ImportError, PolicyGradient,
                          env_mock, pol_mock, CentralFDEstimator(env_mock))
