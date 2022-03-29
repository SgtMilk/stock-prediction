# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
Contains the train_stock function for training a model.
"""

import torch
from src.data import AggregateDataset
from src.hyperparameters import GAN
from src.model import Net
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
        validation_split=GAN.validation_split,
    )

    # getting our models and net
    model = GAN.model(
        device=GAN.device,
        hidden_dim=GAN.hidden_dim,
        num_layers=GAN.num_dim,
        dropout=GAN.dropout,
    )
    optimizer = GAN.optimizer(model.parameters(), lr=GAN.learning_rate, betas=(0.5, 0.999))

    net = Net(
        GAN.device,
        optimizer,
        GAN.loss,
        model,
        dataset,
    )

    # training and evaluating our model
    net.train(GAN.epochs, verbosity_interval=5)

    net.evaluate()
