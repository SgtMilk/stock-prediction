# Copyright (c) 2021 Alix Routhier-Lalonde. Licence included in root of package.

import numpy as np
from .Dataset import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch
import copy


class AggregateDataset:
    """
    This class is to train the model with multiple stock codes
    """

    def __init__(self, codes, interval: int, num_days: int = 50, y_flag=False, no_download=False):
        """
        Will create and aggregate all datasets
        :param codes: an array of stock codes
        :param interval: the interval of days between last real value and prediction
        :param num_days: the number of look back days
        :param y_flag: True if we don't ask to overwrite
        """
        self.interval = interval
        self.num_days = num_days
        self.datasets = []

        for code in codes:
            dataset = Dataset(code, interval, num_days=num_days,
                              y_flag=y_flag, no_download=no_download)
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
        self.normalizer.fit(self.y_unscaled.reshape(-1, 1))

    def get_train(self, split: float = 0.1):
        """
        getter for the training dataset
        :param split: percentage of test data in float format
        :return: None if there is no data, otherwise the x and y train data
        """

        if self.x.any() is None or self.y.any() is None:
            return None
        n = int(self.x.shape[0] * split)
        return self.x[n:], self.y[n:]

    def get_test(self, split: float = 0.1):
        """
        getter for the testing dataset
        :param split: percentage of test data in float format
        :return: None if there is no data, otherwise the x and y_unscaled testing data
        """

        if self.x.any() is None or self.y.any() is None or self.y_unscaled.any() is None:
            print(self.x)
            print(self.y)
            print(self.y_unscaled)
            return None
        n = int(self.x.shape[0] * split)
        return self.x[:n], self.y[:n], self.y_unscaled[:n]

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
        gpu = torch.cuda.is_available()
        if not torch.is_tensor(self.x) and self.x is not None:
            self.x = torch.from_numpy(self.x).float()
            if gpu:
                self.x = self.x.to(device='cuda')

        if not torch.is_tensor(self.y) and self.y is not None:
            self.y = torch.from_numpy(self.y).float()
            if gpu:
                self.y = self.y.to(device='cuda')

        if not torch.is_tensor(self.y_unscaled) and self.y_unscaled is not None:
            self.y_unscaled = torch.from_numpy(self.y_unscaled).float()
            if gpu:
                self.y_unscaled = self.y_unscaled.to(device='cuda')

        for dataset in self.datasets:
            dataset.transform_to_torch()

    def inverse_transform(self, y_data):
        """
        Transforms back the data into unscaled data
        :param y_data: the data to turn back in unscaled
        :return: the scaled data
        """
        return np.squeeze(self.normalizer.inverse_transform(y_data.reshape(-1, 1)))
