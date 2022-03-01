# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
This module contains the GAN class, a collection of hyperparameters to train the GAN model
"""

from torch.nn import MSELoss, BCELoss
from torch.optim import Adam
import torch
from src.model import GeneratorV2, Discriminator


class GAN:
    """
    This class contains all the hyperparameters for training the GAN model
    """

    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    no_download = True

    # dataset parameters
    look_back = 100
    pred_length = 30  # change this to 1 if you want to change training mode
    batch_div = 1024

    # model parameters
    generator = GeneratorV2
    discriminator = Discriminator
    hidden_dim = 128
    num_dim = 2
    dropout = 0
    kernel_size = 1

    # training parameters
    epochs = 150
    learning_rate = 0.001
    loss_G = MSELoss(reduction="mean")
    loss_D = BCELoss(reduction="mean")
    optimizer_G = Adam
    optimizer_D = Adam
