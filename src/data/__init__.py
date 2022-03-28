# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
The data module contains everything related to the input data to the model.
It contains the following data classes:
- Dataset
- AggregateDataset
where AggregateDataset is a collection of Datasets.
It also contains helpers such as:
- get_stock_symbols
- Interval (acts as enum)
"""

from .Dataset import Dataset
from .aggregate_dataset import AggregateDataset
from .get_stock_symbols import get_stock_symbols
