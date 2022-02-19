# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
This module contains the get_stock_symbols function, which will return a list of existing
stock symbols.
"""

import os
import pandas as pd
import numpy as np
from src.utils import get_base_path


def get_stock_symbols():
    """
    This function returns a list of valid stock symbols for training.
    :return a list of stock symbols
    """
    destination_folder = os.path.abspath(os.path.join(get_base_path(), "src/data/stock_prices"))
    file = os.path.join(destination_folder, "symbols" + ".csv")

    data = pd.read_csv(file)
    data = np.array(data)
    data = np.squeeze(data)
    return data
