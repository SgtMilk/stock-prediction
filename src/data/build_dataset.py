from typing import List
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

    # scaling
    scaler = MinMaxScaler()
    data_normalised = scaler.fit_transform(data)

    ohlcv_histories_normalised = np.array(
        [data_normalised[i: i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.array(
        [data_normalised[:, 0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.expand_dims(
        next_day_open_values_normalised, -1)

    next_day_open_values = np.array(
        [data_normalised[:, 0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values_normalised, -1)

    y_normaliser = MinMaxScaler()
    y_normaliser.fit(next_day_open_values_normalised)

    assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0]

    test_split = 0.9  # the percent of data to be used for testing
    n = int(ohlcv_histories_normalised.shape[0] * test_split)

    # splitting the dataset up into train and test sets

    ohlcv_train = ohlcv_histories_normalised[:n]
    y_train = next_day_open_values_normalised[:n]

    ohlcv_test = ohlcv_histories_normalised[n:]
    y_test = next_day_open_values[n:]

    unscaled_y_test = next_day_open_values[n:]

    return ohlcv_train, y_train, ohlcv_test, y_test, unscaled_y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, unscaled_y_test = build_dataset('AAPL')
    print('x_train: \n')
    print(x_train)
    print('\ny_train: \n')
    print(y_train)
