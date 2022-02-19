# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
Source module for the ML part of this repo.
Contains the following functions:
- train_stock
- predict_stock
Contains the following sub-modules:
- data
- hyperparameters
- model
- utils
"""

from .train import train_stock
from .predict import predict_stock
