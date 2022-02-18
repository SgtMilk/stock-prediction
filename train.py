# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

import torch
from src.data import Interval, get_stock_symbols
from src import train_stock

if __name__ == "__main__":
    stock_symbols = get_stock_symbols()
    torch.manual_seed(1)
    train_stock(stock_symbols, Interval.daily)
    