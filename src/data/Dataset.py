from src.utils.print_colors import Colors
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

    def __init__(self, code: str, num_days: int = 50, y_flag: bool = False) -> None:
        """
        __init__ initiates the dataset with a mode and a stock code.
        It also downloads the dataset from the past 5 years and stocks it in ./source

        :param code: the stock's code
        :param num_days: the number of days back the data will be formatted to
        """
        self.code = code
        self.num_days = num_days

        # downloading data and putting it in a .csv file
        data = self.download_data(y_flag=y_flag)

        # initializing variables
        self.x_daily = self.y_daily = self.y_unscaled_daily = self.normalizer_daily = self.x_weekly = self.y_weekly \
            = self.y_unscaled_weekly = self.normalizer_weekly = self.x_monthly = self.y_monthly \
            = self.y_unscaled_monthly = self.normalizer_monthly = None

        # processing data
        for i in [Mode.daily, Mode.weekly, Mode.monthly]:
            self.build_dataset(i, data)

    def update_data(self):
        """
        Re-downloads the data and processes it
        """
        # downloading data and putting it in a .csv file
        data = self.download_data()

        # processing data
        for i in [Mode.daily, Mode.weekly, Mode.monthly]:
            self.build_dataset(i, data)

    def download_data(self, y_flag: bool = False):
        """
        download_data downloads all the data from the past 5 years from that stock code and puts it in .csv files

        :param y_flag: defaults to false, will not ask if you want to overwrite older files
        """
        destination_folder = os.path.abspath('./source')
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
        initial_date = current_date - datetime.timedelta(days=365.24 * 5)
        try:
            data = current_stock.get_historical_price_data(start_date=str(initial_date),
                                                           end_date=str(
                                                               current_date),
                                                           time_interval='daily')
        except OSError:
            print(Colors.FAIL + "\nCould not download data for " +
                  self.code + " (it probably doesn't exist" + Colors.ENDC)
            return None

        prices = data[self.code]['prices']

        with open(destination, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writeheader()
            for price in prices:
                writer.writerow(price)
        print(Colors.OKGREEN + "Data Downloaded!" + '\033[0m' + Colors.ENDC)
        return prices

    def build_dataset(self, mode: int, data=None):
        """
        build dataset will clean up the data according to a mode

        :param data: optional, if it is None, it will fetch data in the according .csv file
        :param mode: Mode.daily, Mode.weekly, Mode.monthly
        :return: x_train, y_train, y_unscaled_train, x_test, y_test, y_unscaled_test, normalizer
        """
        if data is None:
            destination_folder = os.path.abspath('./source')
            file = os.path.join(destination_folder, self.code + '.csv')

            if not os.path.exists(file):
                print(Colors.FAIL + "Data has not been downloaded for this stock code" + Colors.ENDC)
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
        x_data = np.array([data_normalised[i + mode: i + mode + self.num_days]
                           for i in range(len(data_normalised) - self.num_days - (mode - 1))])

        # array of arrays of the next 7 day's data scaled
        y_data = np.array([data_normalised[i: i + mode, 3]
                           for i in range(len(data_normalised) - self.num_days - (mode - 1))])
        # array of arrays of the next 7 day's data unscaled
        y_data_unscaled = np.array([data[i: i + mode, 3]
                                    for i in range(len(data) - self.num_days - (mode - 1))])

        assert x_data.shape[0] == y_data.shape[0]

        normalizer = MinMaxScaler()
        normalizer.fit(y_data_unscaled)

        # setting class variables
        if mode == Mode.daily:
            self.x_daily, self.y_daily, self.y_unscaled_daily, self.normalizer_daily \
                = x_data, y_data, y_data_unscaled, normalizer
        elif mode == Mode.weekly:
            self.x_weekly, self.y_weekly, self.y_unscaled_weekly, self.normalizer_weekly \
                = x_data, y_data, y_data_unscaled, normalizer
        elif mode == Mode.monthly:
            self.x_monthly, self.y_monthly, self.y_unscaled_monthly, self.normalizer_monthly \
                = x_data, y_data, y_data_unscaled, normalizer

        return x_data, y_data, y_data_unscaled, normalizer

    def get_train(self, mode: int, split: float = 0.1):
        """
        getter for the training dataset
        :param mode: Mode.daily, Mode.weekly, Mode.monthly
        :param split: percentage of test data in float format
        :return: None if there is no data, otherwise the x and y train data
        """
        if mode == Mode.daily:
            x = self.x_daily
            y = self.y_daily
        elif mode == Mode.weekly:
            x = self.x_weekly
            y = self.y_weekly
        elif mode == Mode.monthly:
            x = self.x_monthly
            y = self.y_monthly
        else:
            return None

        if x or y is None:
            return None
        n = int(x.shape[0] * split)
        return x[n:], y[n:]

    def get_test(self, mode: int, split: float = 0.1):
        """
        getter for the testing dataset
        :param mode: Mode.daily, Mode.weekly, Mode.monthly
        :param split: percentage of test data in float format
        :return: None if there is no data, otherwise the x and y_unscaled testing data
        """
        if mode == Mode.daily:
            x = self.x_daily
            y_unscaled = self.y_daily
        elif mode == Mode.weekly:
            x = self.x_weekly
            y_unscaled = self.y_weekly
        elif mode == Mode.monthly:
            x = self.x_monthly
            y_unscaled = self.y_monthly
        else:
            return None

        if x or y_unscaled is None:
            return None
        n = int(x.shape[0] * split)
        return x[:n], y_unscaled[:n]

    def get_normalizer(self, mode: int):
        if mode == Mode.daily:
            return self.normalizer_daily
        elif mode == Mode.weekly:
            return self.normalizer_weekly
        elif mode == Mode.monthly:
            return self.normalizer_monthly
        else:
            return None
