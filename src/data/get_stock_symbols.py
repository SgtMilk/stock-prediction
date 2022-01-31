# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

from src.utils import get_base_path
import os
import pandas as pd
import numpy as np


def get_stock_symbols():
    destination_folder = os.path.abspath(
        os.path.join(get_base_path(), 'src/data/stock_prices'))
    file = os.path.join(destination_folder, 'symbols' + '.csv')

    data = pd.read_csv(file)
    data = np.array(data)
    data = np.squeeze(data)
    return data
