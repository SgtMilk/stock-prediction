# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
This module contains the GAN class, a collection of hyperparameters to train the GAN model
"""

from torch.nn import MSELoss, BCELoss
from torch.optim import Adam
import torch
from src.model import RNN


class GAN:
    """
    This class contains all the hyperparameters for training the GAN model
    """

    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    no_download = True

    look_back = 100

    # model parameters
    model = RNN
    hidden_dim = 256
    num_dim = 4
    dropout = 0.4
    validation_split = 0.1

    # training parameters
    epochs = 150
    learning_rate = 0.0001
    loss = MSELoss(reduction="mean")
    optimizer = Adam
