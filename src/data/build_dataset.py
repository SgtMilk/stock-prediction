from typing import List
from numpy.lib.function_base import append
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np

destination_folder = os.path.abspath('./src/data/source')


def build_dataset(code: str, history_points: int = 50):
    file = os.path.join(destination_folder, code + '.csv')

    if not os.path.exists(file):
        print("Data has not been downloaded for this stock code")
        return

    data = pd.read_csv(file)
    del data['date']
    del data['formatted_date']

    data = np.array(data[::-1])

    # scaling
    scaler = MinMaxScaler()
    data_normalised = scaler.fit_transform(data)

    # array of arrays of the last 50 day's data
    x_data = np.array([data_normalised[i + 1: i + 1 + history_points]
                      for i in range(len(data_normalised) - history_points)])

    # array of the next day's data scaled
    y_data = np.array([data_normalised[i]
                      for i in range(len(data_normalised) - history_points)])
    # array of the next day's data unscaled
    y_data_unscaled = np.array([data[i]
                               for i in range(len(data) - history_points)])

    assert x_data.shape[0] == y_data.shape[0]

    test_split = 0.1  # the percent of data to be used for testing
    n = int(x_data.shape[0] * test_split)

    normalizer = MinMaxScaler()
    normalizer.fit(y_data_unscaled)

    # splitting the dataset up into train and test sets

    x_train = x_data[n:]
    y_train = y_data[n:]
    y_unscaled_train = y_data_unscaled[n:]

    x_test = x_data[:n]
    y_test = y_data[:n]
    y_unscaled_test = y_data_unscaled[:n]

    return x_train, y_train, y_unscaled_train, x_test, y_test, y_unscaled_test, normalizer


if __name__ == '__main__':
    x_train, y_train, y_unscaled_train, x_test, y_test, unscaled_y_test, normalizer = build_dataset(
        'AAPL')
    print('x_train shape: ' + str(x_train.shape))
    print('y_train shape: ' + str(y_train.shape))
    print('y_unscaled_train shape: ' + str(y_unscaled_train.shape))
    print('x_test shape: ' + str(x_test.shape))
    print('y_test shape: ' + str(y_test.shape))
    print('unscaled_y_test shape: ' + str(unscaled_y_test.shape))
