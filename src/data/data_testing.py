import unittest
from Dataset import Dataset
import warnings
import numpy as np


def ignore_deprecation_warnings(test_func):
    """
    because yahoo financials has a deprecation warning
    """
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            test_func(self, *args, **kwargs)

    return do_test


class TestStringMethods(unittest.TestCase):
    """
    Testing class for the Dataset class
    """

    @ignore_deprecation_warnings
    def test_shape(self):
        """
        Tests if the data has all the right shape
        """
        num_days = 50
        dataset = Dataset("ARL", num_days, True)

        # x testing
        self.assertEqual(dataset.x_daily.shape[1:], dataset.x_weekly.shape[1:])
        self.assertEqual(dataset.x_daily.shape[1:], dataset.x_monthly.shape[1:])
        self.assertEqual(dataset.x_daily.shape[1:], (num_days, 6))
        self.assertEqual(dataset.x_daily.shape[0], dataset.x_weekly.shape[0] + 4)
        self.assertEqual(dataset.x_daily.shape[0], dataset.x_monthly.shape[0] + 21)

        # y testing
        self.assertEqual(dataset.y_daily.shape, dataset.y_unscaled_daily.shape)
        self.assertEqual(dataset.y_daily.shape, (dataset.x_daily.shape[0], 1))

        self.assertEqual(dataset.y_weekly.shape, dataset.y_unscaled_weekly.shape)
        self.assertEqual(dataset.y_weekly.shape, (dataset.x_weekly.shape[0], 5))

        self.assertEqual(dataset.y_monthly.shape, dataset.y_unscaled_monthly.shape)
        self.assertEqual(dataset.y_monthly.shape, (dataset.x_monthly.shape[0], 22))

    @ignore_deprecation_warnings
    def test_normalizer(self):
        """
        Tests if the normalizer works as expected
        """
        dataset = Dataset("ARL", 50, True)

        self.assertIsNotNone(dataset.normalizer_daily)
        self.assertIsNotNone(dataset.normalizer_weekly)
        self.assertIsNotNone(dataset.normalizer_monthly)

        inverse_daily = dataset.normalizer_daily.inverse_transform(dataset.y_daily)
        inverse_weekly = dataset.normalizer_weekly.inverse_transform(dataset.y_weekly)
        inverse_monthly = dataset.normalizer_monthly.inverse_transform(dataset.y_monthly)

        np.testing.assert_array_almost_equal(inverse_daily, dataset.y_unscaled_daily)
        np.testing.assert_array_almost_equal(inverse_weekly, dataset.y_unscaled_weekly)
        np.testing.assert_array_almost_equal(inverse_monthly, dataset.y_unscaled_monthly)


if __name__ == '__main__':
    unittest.main()
