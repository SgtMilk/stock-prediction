# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from src.utils import progress_bar
from src.utils import Colors
from .Dataset import Dataset


class AggregateDataset:
    """
    This class is to train the model with multiple stock codes
    """

    def __init__(self, device, codes, interval: int, look_back: int = 100, pred_length: int = 30, y_flag=False, no_download=False, batch_div:int = 1, split:float = 0.1):
        """
        Will create and aggregate all datasets
        :param codes: an array of stock codes
        :param interval: the interval of days between last real value and prediction
        :param look_back: the number of look back days
        :param pred_length: the number of predicted days format
        :param y_flag: True if we don't ask to overwrite
        :param batch_div: the number of divisions in the data for the batches
        :param split: the test/train split in the dataset
        """

        if batch_div < 1:
            raise NameError("You cannot have a batch division smaller than 1") from OSError

        self.device = device
        self.interval = interval
        self.look_back = look_back
        self.pred_length = pred_length
        self.batch_div = batch_div
        self.split = split
        self.datasets = []
        total = 0

        for i, code in enumerate(codes):
            dataset = Dataset(code, interval, look_back=look_back, pred_length=pred_length,
                              y_flag=y_flag, no_download=no_download)
            if dataset is None:
                continue
            self.datasets.append(dataset)
            total += dataset.x.shape[0]
            if i == 0:
                self.x = dataset.x
                self.y = dataset.y
                self.y_unscaled = dataset.y_unscaled
            else:
                self.x = np.concatenate((self.x, dataset.x))
                self.y = np.concatenate((self.y, dataset.y))
                self.y_unscaled = np.concatenate((self.y_unscaled, dataset.y_unscaled))
            progress_bar(i + 1, len(codes), suffix="building the dataset...")

        self.normalizer = MinMaxScaler()
        self.normalizer.fit(self.y_unscaled.reshape(-1, 1))

        # transforming to torch tensors
        self.x = torch.from_numpy(self.x).float()
        self.y =torch.from_numpy(self.y).float()
        self.y_unscaled = torch.from_numpy(self.y_unscaled).float()

        # finding the right split
        n_split = int(self.x.shape[0] * (1 - self.split))
        n_split = n_split - n_split % self.batch_div
        self.batch_size = int(n_split / self.batch_div)

        self.x_train, self.y_train, self.y_unscaled_train = self.x[:n_split], self.y[:n_split], self.y_unscaled[:n_split]
        self.x_test, self.y_test, self.y_unscaled_test = self.x[n_split:], self.y[n_split:], self.y_unscaled[n_split:]

        # batching inputs to have cpu -> gpu
        self.x_train = self.x_train.reshape((self.batch_div, self.batch_size, self.x_train.shape[1]))
        self.y_train = self.y_train.reshape((self.batch_div, self.batch_size, self.y_train.shape[1], self.y_train.shape[2]))

    def get_train(self, index: int):
        """
        getter for the training dataset
        :param split: percentage of test data in float format
        :return: None if there is no data, otherwise the x and y train data
        """
        return self.x_train[index].to(device=self.device), self.y_train[index].to(device=self.device)

    def get_test(self):
        """
        getter for the testing dataset
        :param split: percentage of test data in float format
        :return: None if there is no data, otherwise the x and y_unscaled testing data
        """
        return self.x_test.to(device=self.device), self.y_test.to(device=self.device), self.y_unscaled_test.to(device=self.device)

    def inverse_transform(self, y_data):
        """
        Transforms back the data into unscaled data
        :param y_data: the data to turn back in unscaled
        :return: the scaled data
        """
        return self.normalizer.inverse_transform(y_data)
