# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
Contains the Interval and Dataset classes.
"""

import os
import datetime
import csv
import torch
from yahoofinancials import YahooFinancials
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from src.utils import Colors, get_base_path


class Interval:
    """
    Enum equivalent for interval choices
    """

    daily = 1
    weekly = 5
    monthly = 22


class Dataset:
    """Collects data for a stock and organizes it in training or predicting format"""

    def __init__(
        self,
        device,
        code: str,
        interval: int,
        look_back: int = 100,
        pred_length: int = 30,
        y_flag: bool = False,
        no_download=False,
    ) -> None:
        """
        Initiates the dataset.
        It also downloads the dataset from the past 5 years and stocks it in ./source
        :param device: the device to train on (cpu vs cuda)
        :param code: the stock's code
        :param interval: the interval in days between predictions
        :param look_back: the number of days back the x data will be formatted to
        :param pred_length: the number of predicted days format
        :param y_flag: defaults to false, will not ask if you want to overwrite older files
        :param no_download: default to false, will not download any stock data if True
        """
        self.device = device
        self.code = code
        self.interval = interval
        self.look_back = look_back
        self.pred_length = pred_length

        # downloading data and putting it in a .csv file
        if not no_download:
            data = self.download_data(y_flag=y_flag)
        else:
            data = None

        # initializing variables
        self.x_data = self.y_data = self.y_unscaled = self.normalizer = None

        self.build_dataset(data)

    def download_data(self, y_flag: bool = False):
        """
        download_data downloads all the data from the past 5 years from that stock code
        and puts it in .csv files
        :param y_flag: defaults to false, will not ask if you want to overwrite older files
        :return the downloaded stock prices
        """
        destination_folder = os.path.abspath(os.path.join(get_base_path(), "src/data/source"))
        csv_columns = [
            "date",
            "high",
            "low",
            "open",
            "close",
            "volume",
            "adjclose",
            "formatted_date",
        ]

        print(
            Colors.WARNING
            + "If data takes more than 10 seconds to download, ctrl + c will end the forever loop"
            + Colors.ENDC
        )

        destination = os.path.join(destination_folder, self.code + ".csv")
        if y_flag and os.path.exists(destination):
            os.remove(destination)

        if not y_flag and os.path.exists(destination):
            response = input(
                Colors.OKBLUE
                + "would you like to overwrite that file("
                + destination
                + ")? (y/n)"
                + Colors.ENDC
            )
            if response == "y" or response == "yes":
                os.remove(destination)
            else:
                return None

        current_stock = YahooFinancials(self.code)

        current_date = datetime.date.today()
        initial_date = current_date - datetime.timedelta(days=365.24 * 20)
        try:
            data = current_stock.get_historical_price_data(
                start_date=str(initial_date), end_date=str(current_date), time_interval="daily"
            )
        except OSError:
            raise NameError(
                f"\nCould not download data for {self.code} (it probably doesn't exist"
            ) from OSError

        prices = data[self.code]["prices"]

        with open(destination, "w", encoding="utf8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writeheader()
            for price in prices:
                writer.writerow(price)
        print(Colors.OKGREEN + f"{self.code} Data Downloaded!" + "\033[0m" + Colors.ENDC)
        return prices

    def build_dataset(self, data=None):
        """
        build dataset will clean up the data according to an interval length
        :param data: optional, if it is None, it will fetch data in the according .csv file
        :return: x_train, y_train, y_unscaled_train, normalizer
        """
        if data is None:
            destination_folder = os.path.abspath(os.path.join(get_base_path(), "src/data/source"))
            file = os.path.join(destination_folder, self.code + ".csv")

            if not os.path.exists(file):
                print(
                    Colors.FAIL
                    + f"Data has not been downloaded for this stock code ({self.code})"
                    + Colors.ENDC
                )
                return None

            data = pd.read_csv(file)
        else:
            csv_columns = [
                "date",
                "high",
                "low",
                "open",
                "close",
                "volume",
                "adjclose",
                "formatted_date",
            ]
            data = pd.DataFrame(data, columns=csv_columns)

        del data["date"]
        del data["formatted_date"]

        data = np.array(data)

        # scaling
        scaler = MinMaxScaler()
        data_normalised = scaler.fit_transform(data)

        # array of arrays of the last 50 day's data
        x_data = np.array(
            [
                np.array([x[3] for x in data_normalised[i : i + self.look_back]])
                for i in range(
                    len(data_normalised) - self.look_back - self.interval * self.pred_length + 2
                )
            ]
        )
        x_data = np.expand_dims(x_data, axis=2)

        # resulting array of closing prices normalized
        y_data = np.array(
            [
                [
                    np.array([data_normalised[i + self.look_back + self.interval * j - 1, 3]])
                    for j in range(self.pred_length)
                ]
                for i in range(
                    len(data_normalised) - self.look_back - self.interval * self.pred_length + 2
                )
            ]
        )

        # resulting array of closing prices unscaled
        y_data_unscaled = np.array(
            [
                [
                    data[i + self.look_back + self.interval * j - 1, 3]
                    for j in range(self.pred_length)
                ]
                for i in range(len(data) - self.look_back - self.interval * self.pred_length + 2)
            ]
        )

        assert x_data.shape[0] == y_data.shape[0]
        assert y_data.shape[0] == y_data_unscaled.shape[0]
        if y_data.size == 0:
            return None

        normalizer = MinMaxScaler()
        normalizer.fit(y_data_unscaled)

        # setting class variables
        self.x_data = []
        self.y_data = []
        self.y_unscaled = []
        self.normalizer = normalizer

        # removing all NaN values
        for i in range(x_data.shape[0]):
            if (
                np.isnan(np.sum(x_data[i].flatten()))
                or np.isnan(np.sum(y_data[i].flatten()))
                or np.isnan(np.sum(y_data_unscaled[i].flatten()))
            ):
                continue
            self.x_data.append(x_data[i])
            self.y_data.append(y_data[i])
            self.y_unscaled.append(y_data_unscaled[i])

        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)
        self.y_unscaled = np.array(self.y_unscaled)

        return self.x_data, self.y_data, self.y_unscaled, self.normalizer

    def inverse_transform(self, y_data):
        """
        Transforms back the data into unscaled data
        :param y_data: the data to turn back in unscaled
        :return: the scaled data
        """
        return self.normalizer.inverse_transform(y_data)

    def transform_to_torch(self):
        """
        Transforms all the class data to torch tensors
        """
        if not torch.is_tensor(self.x_data) and self.x_data is not None:
            self.x_data = torch.from_numpy(self.x_data).float().to(device=self.device)

        if not torch.is_tensor(self.y_data) and self.y_data is not None:
            self.y_data = torch.from_numpy(self.y_data).float().to(device=self.device)

        if not torch.is_tensor(self.y_unscaled) and self.y_unscaled is not None:
            self.y_unscaled = torch.from_numpy(self.y_unscaled).float().to(device=self.device)
