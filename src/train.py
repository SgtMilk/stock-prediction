# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
Contains the train_stock function for training a model.
"""

import torch
from src.data import AggregateDataset
from src.hyperparameters import GAN
from src.model import Net, init_weights
from src.utils import Colors


def train_stock(codes) -> None:
    """
    Trains a stock prediction model with a downloaded array of stock prices.
    You can adjust the hyperparameters of the training in `./hyperparameters/gan.py`.
    :param codes: array of stock codes
    """

    if torch.cuda.is_available() is False:
        print(Colors.FAIL + "You are not training on a GPU" + Colors.ENDC)
        return

    # getting the data
    dataset = AggregateDataset(
        GAN.device,
        codes,
        y_flag=True,
        no_download=GAN.no_download,
    )

    # getting our models and net
    generator = GAN.generator(
        device=GAN.device,
        hidden_dim=GAN.hidden_dim,
        num_layers=GAN.num_dim,
        dropout=GAN.dropout,
    )
    optimizer_g = GAN.optimizer_G(generator.parameters(), lr=GAN.learning_rate, betas=(0.5, 0.999))
    generator.apply(init_weights)

    discriminator = GAN.discriminator(
        device=GAN.device,
        hidden_dim=GAN.hidden_dim,
        num_layers=GAN.num_dim,
        dropout=GAN.dropout,
    )

    optimizer_d = GAN.optimizer_D(
        discriminator.parameters(), lr=GAN.learning_rate, betas=(0.5, 0.999)
    )
    discriminator.apply(init_weights)

    net = Net(
        GAN.device,
        optimizer_g,
        optimizer_d,
        GAN.loss_G,
        GAN.loss_D,
        generator,
        discriminator,
        dataset,
    )

    # training and evaluating our model
    net.train(GAN.epochs, verbosity_interval=5)
    net.save()

    net.evaluate()
