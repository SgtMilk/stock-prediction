# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
============
train module
============

This module will train a model according to an interval.
Please set the interval on line 13 before running.
If you do not have any cuda-capable gpu on your machine, an error will appear and stop the script.
"""

import os
import torch
from src.data import get_stock_symbols
from src import train_stock

if __name__ == "__main__":
    stock_symbols = get_stock_symbols()
    torch.manual_seed(1)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    train_stock(stock_symbols)
