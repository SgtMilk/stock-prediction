# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

import torch
from src.data import AggregateDataset
from src.hyperparameters import GAN
from src.model import Net, init_weights
from src.utils import Colors


def train_stock(codes, interval: int):
    """
    Trains one stock
    :param codes: array of stock codes
    :param interval: Interval.daily, Interval.weekly, Interval.monthly
    """

    if torch.cuda.is_available() is False:
        print(Colors.FAIL + "You are not training on a GPU" + Colors.ENDC)
        return

    # getting the data
    dataset = AggregateDataset(GAN.device, codes, interval=interval, look_back=GAN.look_back, pred_length=GAN.pred_length, y_flag=True, no_download=True, batch_div = 128)

    # getting our models and net
    generator = GAN.generator(dataset.x.shape[-1], GAN.hidden_dim, GAN.num_dim, GAN.dropout, dataset.y.shape[-2], GAN.kernel_size)
    optimizer_g = GAN.optimizer_G(generator.parameters(), lr=GAN.learning_rate, betas=(0.5, 0.999))
    generator.apply(init_weights)

    discriminator = GAN.discriminator(dataset.y.shape[-2], GAN.hidden_dim, GAN.num_dim, GAN.dropout, GAN.kernel_size)
    optimizer_d = GAN.optimizer_D(discriminator.parameters(), lr=GAN.learning_rate, betas=(0.5, 0.999))
    discriminator.apply(init_weights)

    net = Net(GAN.device, optimizer_g, optimizer_d, GAN.loss, generator, discriminator, dataset)

    # training and evaluating our model
    net.train(GAN.epochs, GAN.patience, verbosity_interval=50)
    net.save()
    # net.evaluate_training()   # only useful if you are not using tensorboard
    net.evaluate(dataset)
