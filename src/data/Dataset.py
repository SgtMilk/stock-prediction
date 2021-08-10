import torch

from src.utils import Colors, get_base_path
from yahoofinancials import YahooFinancials
from sklearn.preprocessing import MinMaxScaler
import os
import datetime
import csv
import pandas as pd
import numpy as np


class Mode:
    daily = 1
    weekly = 5
    monthly = 22


class Dataset:
    """collects data for a stock"""

    def __init__(self, code: str, mode: int, num_days: int = 50, y_flag: bool = False) -> None:
        """
        __init__ initiates the dataset with a mode and a stock code.
        It also downloads the dataset from the past 5 years and stocks it in ./source

        :param code: the stock's code
        :param num_days: the number of days back the data will be formatted to
        """
        self.code = code
        self.mode = mode
        self.num_days = num_days

        # downloading data and putting it in a .csv file
        data = self.download_data(y_flag=y_flag)

        # initializing variables
        self.x = self.y = self.y_unscaled = self.normalizer = None

        self.build_dataset(data)

    def update_data(self):
        """
        Re-downloads the data and processes it
        """
        # downloading data and putting it in a .csv file
        data = self.download_data()

        # processing data
        self.build_dataset(data)

    def download_data(self, y_flag: bool = False):
        """
        download_data downloads all the data from the past 5 years from that stock code and puts it in .csv files

        :param y_flag: defaults to false, will not ask if you want to overwrite older files
        """
        destination_folder = os.path.abspath(os.path.join(get_base_path(), 'src/data/source'))
        csv_columns = ['date', 'high', 'low', 'open',
                       'close', 'volume', 'adjclose', 'formatted_date']

        print(Colors.WARNING + "If data takes more than 10 seconds to download, ctrl + c will end the forever loop"
              + Colors.ENDC)

        destination = os.path.join(destination_folder, self.code + '.csv')
        if y_flag and os.path.exists(destination):
            os.remove(destination)

        if not y_flag and os.path.exists(destination):
            response = input(Colors.OKBLUE +
                             "would you like to overwrite that file(" + destination + ")? (y/n)" + Colors.ENDC)
            if response == 'y' or response == 'yes':
                os.remove(destination)
            else:
                return None

        current_stock = YahooFinancials(self.code)

        current_date = datetime.date.today()
        initial_date = current_date - datetime.timedelta(days=365.24 * 20)
        try:
            data = current_stock.get_historical_price_data(start_date=str(initial_date),
                                                           end_date=str(
                                                               current_date),
                                                           time_interval='daily')
        except OSError:
            raise NameError(f"\nCould not download data for {self.code} (it probably doesn't exist")

        prices = data[self.code]['prices']

        with open(destination, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writeheader()
            for price in prices:
                writer.writerow(price)
        print(Colors.OKGREEN + f"{self.code} Data Downloaded!" + '\033[0m' + Colors.ENDC)
        return prices

    def build_dataset(self, data=None):
        """
        build dataset will clean up the data according to a mode

        :param data: optional, if it is None, it will fetch data in the according .csv file
        :return: x_train, y_train, y_unscaled_train, x_test, y_test, y_unscaled_test, normalizer
        """
        if data is None:
            destination_folder = os.path.abspath(os.path.join(get_base_path(), 'src/data/source'))
            file = os.path.join(destination_folder, self.code + '.csv')

            if not os.path.exists(file):
                print(Colors.FAIL + f"Data has not been downloaded for this stock code ({self.code})" + Colors.ENDC)
                return None

            data = pd.read_csv(file)
        else:
            csv_columns = ['date', 'high', 'low', 'open',
                           'close', 'volume', 'adjclose', 'formatted_date']
            data = pd.DataFrame(data, columns=csv_columns)

        del data['date']
        del data['formatted_date']

        data = np.array(data[::-1])

        # scaling
        scaler = MinMaxScaler()
        data_normalised = scaler.fit_transform(data)

        # array of arrays of the last 50 day's data
        x_data = np.array([data_normalised[i + self.mode: i + self.mode + self.num_days]
                           for i in range(len(data_normalised) - self.num_days - (self.mode - 1))])

        # array of arrays of the next 7 day's data scaled
        y_data = np.array([data_normalised[i: i + self.mode, 3]
                           for i in range(len(data_normalised) - self.num_days - (self.mode - 1))])
        # array of arrays of the next 7 day's data unscaled
        y_data_unscaled = np.array([data[i: i + self.mode, 3]
                                    for i in range(len(data) - self.num_days - (self.mode - 1))])

        assert x_data.shape[0] == y_data.shape[0]

        normalizer = MinMaxScaler()
        normalizer.fit(y_data_unscaled)

        # setting class variables
        self.x, self.y, self.y_unscaled, self.normalizer = x_data, y_data, y_data_unscaled, normalizer

        return x_data, y_data, y_data_unscaled, normalizer

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
