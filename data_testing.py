# Copyright (c) 2021 Alix Routhier-Lalonde. Licence included in root of package.

import unittest
from src.data import Dataset, Interval
import warnings
import numpy as np


def ignore_deprecation_warnings(test_func):
    """
    Ignores deprecation warnings
    """

    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            test_func(self, *args, **kwargs)

    return do_test


class TestDataset(unittest.TestCase):
    """
    Testing class for the Dataset class
    """

    @ignore_deprecation_warnings
    def test_shape(self):
        """
        Tests if the data has all the right shape
        """
        num_days = 50
        dataset_daily = Dataset("ARL", interval=Interval.daily, num_days=num_days, y_flag=True)
        dataset_weekly = Dataset("ARL", interval=Interval.weekly, num_days=num_days, y_flag=True)
        dataset_monthly = Dataset("ARL", interval=Interval.monthly, num_days=num_days, y_flag=True)

        # x testing
        self.assertEqual(dataset_daily.x.shape[1:], dataset_weekly.x.shape[1:])
        self.assertEqual(dataset_daily.x.shape[1:], dataset_monthly.x.shape[1:])
        self.assertEqual(dataset_daily.x.shape[1:], (num_days, 1))
        self.assertEqual(dataset_daily.x.shape[0], dataset_weekly.x.shape[0] + 4)
        self.assertEqual(dataset_daily.x.shape[0], dataset_monthly.x.shape[0] + 21)

        # y testing
        self.assertEqual(dataset_daily.y.shape, dataset_daily.y_unscaled.shape)
        self.assertEqual(dataset_daily.y.shape, (dataset_daily.x.shape[0],))

        self.assertEqual(dataset_weekly.y.shape, dataset_weekly.y_unscaled.shape)
        self.assertEqual(dataset_weekly.y.shape, (dataset_weekly.x.shape[0],))

        self.assertEqual(dataset_monthly.y.shape, dataset_monthly.y_unscaled.shape)
        self.assertEqual(dataset_monthly.y.shape, (dataset_monthly.x.shape[0],))

        assert dataset_daily.x.any() is not None and dataset_daily.y.any() is not None
        assert dataset_weekly.x.any() is not None and dataset_weekly.y.any() is not None
        assert dataset_monthly.x.any() is not None and dataset_monthly.y.any() is not None


    @ignore_deprecation_warnings
    def test_normalizer(self):
        """
        Tests if the normalizer works as expected
        """
        dataset_daily = Dataset("ARL", interval=Interval.daily, num_days=50, y_flag=True)
        dataset_weekly = Dataset("ARL", interval=Interval.weekly, num_days=50, y_flag=True)

        self.assertIsNotNone(dataset_daily.normalizer)
        self.assertIsNotNone(dataset_weekly.normalizer)

        inverse_daily = dataset_daily.inverse_transform(dataset_daily.y)
        inverse_weekly = dataset_weekly.inverse_transform(dataset_weekly.y)

        np.testing.assert_array_almost_equal(inverse_daily, dataset_daily.y_unscaled)
        np.testing.assert_array_almost_equal(inverse_weekly, dataset_weekly.y_unscaled)

    @ignore_deprecation_warnings
    def test_interval(self):
        """
        Tests if the interval works properly
        """
        num_days = 50
        dataset_daily = Dataset("ARL", interval=Interval.daily, num_days=num_days, y_flag=True)
        dataset_weekly = Dataset("ARL", interval=Interval.weekly, num_days=num_days, y_flag=True)
        dataset_monthly = Dataset("ARL", interval=Interval.monthly, num_days=num_days, y_flag=True)

        np.testing.assert_array_equal(dataset_daily.y[4:], dataset_weekly.y)
        np.testing.assert_array_equal(dataset_daily.y[21:], dataset_monthly.y)

if __name__ == '__main__':
    unittest.main()
