import os
import pandas as pd
import numpy as np


def get_stock_symbols():
    destination_folder = os.path.abspath('./data/stock_prices')
    file = os.path.join(destination_folder, 'symbols' + '.csv')

    data = pd.read_csv(file)
    data = np.array(data)
    data = np.squeeze(data)
    return data
