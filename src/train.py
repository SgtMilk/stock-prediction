# Copyright (c) 2021 Alix Routhier-Lalonde. Licence included in root of package.

from src.data import AggregateDataset
from src.hyperparameters import Train
from src.model import Net


def train_stock(codes, interval: int):
    """
    Trains one stock
    :param codes: array of stock codes
    :param interval: Interval.daily, Interval.weekly, Interval.monthly
    """
    # getting the data
    dataset = AggregateDataset(codes, interval=interval, y_flag=True, no_download=True)
    dataset.transform_to_torch()

    # getting our model and net
    model = Train.model(dataset.x.shape[-1], Train.hidden_dim, Train.num_dim, Train.dropout, 1)
    net = Net(Train.optimizer(model.parameters(), lr=Train.learning_rate),
              Train.loss(reduction='mean'), model, dataset)

    # training and evaluating our model
    net.train(Train.epochs, dataset, Train.validation_split, Train.patience)
    net.evaluate_training()
    net.evaluate(dataset)
    return net.model
