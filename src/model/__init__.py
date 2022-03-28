# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
The model module contains everything related to the Pytorch model and training it.
It contains the following models:
- Generator
- Discriminator
It also contains the Net class, which will train the models, as well as the init_weights function,
which initializes the weights of a model.
"""

from .Net import Net
from .generatorv1 import GeneratorV1
from .generatorv2 import GeneratorV2
from .generatorv3 import GeneratorV3
from .discriminator_rnn import DiscriminatorRNN
from .generator_rnn import GeneratorRNN
from .init_weights import init_weights
