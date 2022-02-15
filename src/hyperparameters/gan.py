# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

from torch.nn import MSELoss
from torch.optim import Adam
import torch
from src.model import Generator, Discriminator


class GAN:
    """
    This class contains all the hyperparameters for training the GAN model
    """
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    epochs = 100
    patience = epochs
    learning_rate = 0.0001
    loss = MSELoss(reduction='mean')
    optimizer_G = Adam
    optimizer_D = Adam
