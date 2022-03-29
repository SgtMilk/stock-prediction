# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
Contains the AggregateDataset class, a class for training the model
"""

from sklearn.preprocessing import MinMaxScaler
import torch
from torch.nn.utils.rnn import pad_sequence
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
        y_flag=False,
        no_download=False,
        validation_split: float = 0.1,
        test_split: float = 0.1,
    ):
        """
        Will create and aggregate all datasets
        :param device: the device to train on (cpu vs cuda)
        :param codes: an array of stock codes
        :param y_flag: True if we don't ask to overwrite
        :param batch_div: the number of batches in the data
        :param validation_split: the test split in the dataset
        :param test_split: the test split in the dataset
        """

        self.device = device
        self.validation_split = validation_split
        self.test_split = test_split
        self.datasets = []
        self.x_data = []
        self.y_data = []
        self.y_unscaled = []

        for i, code in enumerate(codes):
            dataset = Dataset(
                device,
                code,
                y_flag=y_flag,
                no_download=no_download,
            )
            if dataset is None:
                continue
            if dataset.x_data is None or dataset.y_data is None or dataset.y_unscaled is None:
                continue
            if dataset.x_data.ndim != 1 or dataset.y_data.ndim != 1 or dataset.y_unscaled.ndim != 1:
                continue

            self.datasets.append(dataset)

            self.x_data.append(torch.from_numpy(dataset.x_data))
            self.y_data.append(torch.from_numpy(dataset.y_data))
            self.y_unscaled.append(torch.from_numpy(dataset.y_unscaled))

            progress_bar(i + 1, len(codes), suffix="building the dataset...")

        # padding the sequences
        self.x_data = pad_sequence(self.x_data).float()
        self.y_data = pad_sequence(self.y_data).float()
        self.y_unscaled = pad_sequence(self.y_unscaled).float()

        self.normalizer = MinMaxScaler()
        self.normalizer.fit(self.y_unscaled.numpy())

        # finding the right split
        v_split = int(self.x_data.shape[0] * (1 - self.validation_split - self.test_split))
        t_split = int(self.x_data.shape[0] * (1 - self.test_split))

        self.x_train, self.y_train, self.y_unscaled_train = (
            self.x_data[:v_split].permute((1, 0)),
            self.y_data[:v_split].permute((1, 0)),
            self.y_unscaled[:v_split].permute((1, 0)),
        )
        self.x_validation, self.y_validation, self.y_unscaled_validation = (
            self.x_data[v_split:t_split].permute((1, 0)),
            self.y_data[v_split:t_split].permute((1, 0)),
            self.y_unscaled[v_split:t_split].permute((1, 0)),
        )
        self.x_test, self.y_test, self.y_unscaled_test = (
            self.x_data[t_split:].permute((1, 0)),
            self.y_data[t_split:].permute((1, 0)),
            self.y_unscaled[t_split:].permute((1, 0)),
        )

        self.num_train_batches = self.x_train.shape[0]
        self.num_validation_batches = self.x_validation.shape[0]
        self.num_test_batches = self.x_test.shape[0]

    def get_train(self, index: int):
        """
        getter for part of the training dataset.
        Will return the data on the asked device
        :param index: the index in the x_train array of data
        :return: (x, y) train data
        """
        return self.x_train[index].to(device=self.device), self.y_train[index].to(
            device=self.device
        )

    def get_validation(self, index: int):
        """
        getter for part of the validation dataset.
        Will return the data on the asked device
        :param index: the index in the x_train array of data
        :return: (x, y) train data
        """
        return self.x_validation[index].to(device=self.device), self.y_validation[index].to(
            device=self.device
        )

    def get_test(self, index: int):
        """
        getter for part of the testing dataset.
        Will return the data on the asked device
        :param index: the index in the x_test array of data
        :return: (x, y_unscaled) test data
        """
        return (
            self.x_test[index].to(device=self.device),
            self.y_unscaled_test[index].to(device=self.device),
        )

    def inverse_transform(self, y_data):
        """
        Transforms back the data into unscaled data
        :param y_data: the data to turn back in unscaled
        :return: the scaled data
        """
        return self.normalizer.inverse_transform(y_data)
