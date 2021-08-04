import unittest
from src.data import Dataset, Mode
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
        dataset_daily = Dataset("ARL", mode=Mode.daily, num_days=num_days, y_flag=True)
        dataset_weekly = Dataset("ARL", mode=Mode.weekly, num_days=num_days, y_flag=True)
        dataset_monthly = Dataset("ARL", mode=Mode.monthly, num_days=num_days, y_flag=True)

        # x testing
        self.assertEqual(dataset_daily.x.shape[1:], dataset_weekly.x.shape[1:])
        self.assertEqual(dataset_daily.x.shape[1:], dataset_monthly.x.shape[1:])
        self.assertEqual(dataset_daily.x.shape[1:], (num_days, 6))
        self.assertEqual(dataset_daily.x.shape[0], dataset_weekly.x.shape[0] + 4)
        self.assertEqual(dataset_daily.x.shape[0], dataset_monthly.x.shape[0] + 21)

        # y testing
        self.assertEqual(dataset_daily.y.shape, dataset_daily.y_unscaled.shape)
        self.assertEqual(dataset_daily.y.shape, (dataset_daily.x.shape[0], 1))

        self.assertEqual(dataset_weekly.y.shape, dataset_weekly.y_unscaled.shape)
        self.assertEqual(dataset_weekly.y.shape, (dataset_weekly.x.shape[0], 5))

        self.assertEqual(dataset_monthly.y.shape, dataset_monthly.y_unscaled.shape)
        self.assertEqual(dataset_monthly.y.shape, (dataset_monthly.x.shape[0], 22))

    @ignore_deprecation_warnings
    def test_normalizer(self):
        """
        Tests if the normalizer works as expected
        """
        dataset_daily = Dataset("ARL", mode=Mode.daily, num_days=50, y_flag=True)
        dataset_weekly = Dataset("ARL", mode=Mode.weekly, num_days=50, y_flag=True)

        self.assertIsNotNone(dataset_daily.normalizer)
        self.assertIsNotNone(dataset_weekly.normalizer)

        inverse_daily = dataset_daily.normalizer.inverse_transform(dataset_daily.y)
        inverse_weekly = dataset_weekly.normalizer.inverse_transform(dataset_weekly.y)

        np.testing.assert_array_almost_equal(inverse_daily, dataset_daily.y_unscaled)
        np.testing.assert_array_almost_equal(inverse_weekly, dataset_weekly.y_unscaled)


if __name__ == '__main__':
    unittest.main()
