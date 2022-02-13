# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

from torch.nn import MSELoss
from torch.optim import Adam
from src.model import GRUModel


class Train:
    """
    This class contains all the hyperparameters for training the model
    """
    # model parameters
    model = GRUModel
    hidden_dim = 256
    num_dim = 2
    dropout = 0

    # training parameters
    epochs = 1000
    patience = epochs
    learning_rate = 0.001
    validation_split = 0.1
    loss = MSELoss
    optimizer = Adam
