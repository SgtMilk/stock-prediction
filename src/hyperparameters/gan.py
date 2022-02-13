# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

from torch.nn import BCELoss
from torch.optim import Adam
from src.model import Generator, Discriminator


class GAN:
    """
    This class contains all the hyperparameters for training the GAN model
    """

    # dataset parameters
    look_back = 100
    pred_length = 30

    # model parameters
    generator = Generator
    discriminator = Discriminator
    hidden_dim = 256
    num_dim = 2
    dropout = 0
    kernel_size = 1

    # training parameters
    epochs = 1000
    patience = epochs
    learning_rate = 0.001
    loss = BCELoss(reduction='mean')
    optimizer_G = Adam
    optimizer_D = Adam
