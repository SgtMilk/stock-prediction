# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
===================
data_testing module
===================

This module will run all the unit tests in this repo.
"""

import warnings
import unittest
import numpy as np
import torch
from src.data import Dataset, Interval


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
        dataset_daily = Dataset(torch.device("cpu"), "ARL", interval=Interval.daily, y_flag=True)
        dataset_weekly = Dataset(torch.device("cpu"), "ARL", interval=Interval.weekly, y_flag=True)
        dataset_monthly = Dataset(
            torch.device("cpu"), "ARL", interval=Interval.monthly, y_flag=True
        )

        # x testing
        self.assertEqual(dataset_daily.x_data.shape[1:], dataset_weekly.x_data.shape[1:])
        self.assertEqual(dataset_daily.x_data.shape[1:], dataset_monthly.x_data.shape[1:])
        self.assertEqual(dataset_daily.x_data.shape[1:], (100, 1))

        # y testing
        self.assertEqual(dataset_daily.y_data.shape[:2], dataset_daily.y_unscaled.shape[:2])
        self.assertEqual(dataset_daily.y_data.shape[0], dataset_daily.x_data.shape[0])

        self.assertEqual(dataset_weekly.y_data.shape[:2], dataset_weekly.y_unscaled.shape[:2])
        self.assertEqual(dataset_weekly.y_data.shape[0], dataset_weekly.x_data.shape[0])

        self.assertEqual(dataset_monthly.y_data.shape[:2], dataset_monthly.y_unscaled.shape[:2])
        self.assertEqual(dataset_monthly.y_data.shape[0], dataset_monthly.x_data.shape[0])

        assert dataset_daily.x_data.any() is not None and dataset_daily.y_data.any() is not None
        assert dataset_weekly.x_data.any() is not None and dataset_weekly.y_data.any() is not None
        assert dataset_monthly.x_data.any() is not None and dataset_monthly.y_data.any() is not None

    @ignore_deprecation_warnings
    def test_normalizer(self):
        """
        Tests if the normalizer works as expected
        """
        dataset_daily = Dataset(torch.device("cpu"), "ARL", interval=Interval.daily, y_flag=True)
        dataset_weekly = Dataset(torch.device("cpu"), "ARL", interval=Interval.weekly, y_flag=True)

        self.assertIsNotNone(dataset_daily.normalizer)
        self.assertIsNotNone(dataset_weekly.normalizer)

        inverse_daily = dataset_daily.inverse_transform(dataset_daily.y_data.squeeze())
        inverse_weekly = dataset_weekly.inverse_transform(dataset_weekly.y_data.squeeze())

        np.testing.assert_array_almost_equal(inverse_daily, dataset_daily.y_unscaled)
        np.testing.assert_array_almost_equal(inverse_weekly, dataset_weekly.y_unscaled)


if __name__ == "__main__":
    unittest.main()
