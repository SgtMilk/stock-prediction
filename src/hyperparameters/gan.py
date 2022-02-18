# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

from torch.nn import MSELoss, BCELoss
from torch.optim import Adam
import torch
from src.model import Generator, Discriminator


class GAN:
    """
    This class contains all the hyperparameters for training the GAN model
    """
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    no_download = True

    # dataset parameters
    look_back = 100
    pred_length = 30
    batch_div = 256

    # model parameters
    generator = Generator
    discriminator = Discriminator
    hidden_dim = 256
    num_dim = 2
    dropout = 0
    kernel_size = 1

    # training parameters
    epochs = 20
    patience = epochs
    learning_rate = 0.0001
    loss_G = MSELoss(reduction='mean')
    loss_D = BCELoss(reduction='mean')
    optimizer_G = Adam
    optimizer_D = Adam
