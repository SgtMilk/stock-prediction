# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
The model module contains everything related to the Pytorch model and training it.
It contains the following models:
- Generator
- Discriminator
It also contains the Net class, which will train the models, as well as the init_weights function,
which initializes the weights of a model.
"""

from .net import Net
from .generator import Generator
from .discriminator import Discriminator
from .init_weights import init_weights
