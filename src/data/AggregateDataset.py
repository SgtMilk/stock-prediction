import numpy as np
from .Dataset import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch


class AggregateDataset:
    """
    This class is to train the model with multiple stock codes
    """
    def __init__(self, codes, mode: int, num_days: int = 50, y_flag=False):
        """
        Will create and aggregate all datasets
        :param codes: an array of stock codes
        :param mode: Mode.daily, Mode.weekly, Mode.monthly
        :param num_days: the number of look back days
        :param y_flag: True if we don't ask to overwrite
        """
        self.mode = mode
        self.num_days = num_days
        self.datasets = []

        for code in codes:
            dataset = Dataset(code, mode, num_days=num_days, y_flag=y_flag)
            self.datasets.append(dataset)
            if code == codes[0]:
                self.x = dataset.x
                self.y = dataset.y
                self.y_unscaled = dataset.y_unscaled
            else:
                np.concatenate((self.x, dataset.x))
                np.concatenate((self.y, dataset.y))
                np.concatenate((self.y_unscaled, dataset.y_unscaled))

        self.normalizer = MinMaxScaler()
        self.normalizer.fit(self.y_unscaled)

    def get_train(self, split: float = 0.1):
        """
        getter for the training dataset
        :param split: percentage of test data in float format
        :return: None if there is no data, otherwise the x and y train data
        """

        if self.x.all() or self.y.all() is None:
            return None
        n = int(self.x.shape[0] * split)
        return self.x[n:], self.y[n:]

    def get_test(self, split: float = 0.1):
        """
        getter for the testing dataset
        :param split: percentage of test data in float format
        :return: None if there is no data, otherwise the x and y_unscaled testing data
        """

        if self.x.all() or self.y_unscaled.all() is None:
            return None
        n = int(self.x.shape[0] * split)
        return self.x[:n], self.y_unscaled[:n]

    def transform_to_numpy(self):
        """
        Transforms all the class data to numpy arrays
        """
        if torch.is_tensor(self.x):
            self.x = self.x.numpy()
        if torch.is_tensor(self.y):
            self.y = self.y.numpy()
        if torch.is_tensor(self.y_unscaled):
            self.y_unscaled = self.y_unscaled.numpy()

        for dataset in self.datasets:
            dataset.transform_to_numpy()

    def transform_to_torch(self):
        """
        Transforms all the class data to torch tensors
        """
        if not torch.is_tensor(self.x) and self.x is not None:
            self.x = torch.from_numpy(self.x).float()

        if not torch.is_tensor(self.y) and self.y is not None:
            self.y = torch.from_numpy(self.y).float()

        if not torch.is_tensor(self.y_unscaled) and self.y_unscaled is not None:
            self.y_unscaled = torch.from_numpy(self.y_unscaled).float()

        for dataset in self.datasets:
            dataset.transform_to_torch()
