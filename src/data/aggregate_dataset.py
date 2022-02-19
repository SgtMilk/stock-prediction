# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
Contains the AggregateDataset class, a class for training the model
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from src.utils import progress_bar
from src.data import Dataset


class AggregateDataset:
    """
    This class is to train the model with multiple stock codes.
    It will act as a generator, by shifting data from main memory to gpu memory.
    """

    def __init__(
        self,
        device,
        codes,
        interval: int,
        look_back: int = 100,
        pred_length: int = 30,
        y_flag=False,
        no_download=False,
        batch_div: int = 1,
        split: float = 0.1,
    ):
        """
        Will create and aggregate all datasets
        :param device: the device to train on (cpu vs cuda)
        :param codes: an array of stock codes
        :param interval: the interval of days between last real value and prediction
        :param look_back: the number of look back days
        :param pred_length: the number of predicted days format
        :param y_flag: True if we don't ask to overwrite
        :param batch_div: the number of batches in the data
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

        for i, code in enumerate(codes):
            dataset = Dataset(
                device,
                code,
                interval,
                look_back=look_back,
                pred_length=pred_length,
                y_flag=y_flag,
                no_download=no_download,
            )
            if dataset is None:
                continue
            if dataset.x_data is None or dataset.y_data is None or dataset.y_unscaled is None:
                continue
            if dataset.x_data.ndim != 3 or dataset.y_data.ndim != 3 or dataset.y_unscaled.ndim != 2:
                continue

            self.datasets.append(dataset)
            if i == 0:
                self.x_data = dataset.x_data
                self.y_data = dataset.y_data
                self.y_unscaled = dataset.y_unscaled
            else:
                self.x_data = np.concatenate((self.x_data, dataset.x_data))
                self.y_data = np.concatenate((self.y_data, dataset.y_data))
                self.y_unscaled = np.concatenate((self.y_unscaled, dataset.y_unscaled))
            progress_bar(i + 1, len(codes), suffix="building the dataset...")

        self.normalizer = MinMaxScaler()
        self.normalizer.fit(self.y_unscaled.reshape(-1, 1))

        # transforming to torch tensors
        self.batch_size = int(self.x_data.shape[0] / batch_div)
        self.x_data = torch.from_numpy(self.x_data[: batch_div * self.batch_size]).float()
        self.y_data = torch.from_numpy(self.y_data[: batch_div * self.batch_size]).float()
        self.y_unscaled = torch.from_numpy(self.y_unscaled[: batch_div * self.batch_size]).float()

        # reshaping into batches
        self.x_data = self.x_data.reshape(
            (self.batch_div, self.batch_size, self.x_data.shape[1], self.x_data.shape[2])
        )
        self.y_data = self.y_data.reshape(
            (self.batch_div, self.batch_size, self.y_data.shape[1], self.y_data.shape[2])
        )
        self.y_unscaled = self.y_unscaled.reshape(
            (self.batch_div, self.batch_size, self.y_unscaled.shape[1])
        )

        # finding the right split
        n_split = int(self.x_data.shape[0] * (1 - self.split))

        self.x_train, self.y_train, self.y_unscaled_train = (
            self.x_data[:n_split],
            self.y_data[:n_split],
            self.y_unscaled[:n_split],
        )
        self.x_test, self.y_test, self.y_unscaled_test = (
            self.x_data[n_split:],
            self.y_data[n_split:],
            self.y_unscaled[n_split:],
        )

        self.num_train_batches = n_split
        self.num_test_batches = self.batch_div - n_split

    def get_train(self, index: int):
        """
        getter for part of the training dataset. Acts as a generator.
        Will return the data on the asked device
        :param index: the index in the x_train array of data
        :return: (x, y) train data
        """
        return self.x_train[index].to(device=self.device), self.y_train[index].to(
            device=self.device
        )

    def get_test(self, index: int):
        """
        getter for part of the testing dataset. Acts as a generator.
        Will return the data on the asked device
        :param index: the index in the x_test array of data
        :return: (x, y) test data
        """
        return (
            self.x_test[index].to(device=self.device),
            self.y_test[index].to(device=self.device),
            self.y_unscaled_test[index].to(device=self.device),
        )

    def inverse_transform(self, y_data):
        """
        Transforms back the data into unscaled data
        :param y_data: the data to turn back in unscaled
        :return: the scaled data
        """
        return self.normalizer.inverse_transform(y_data)
