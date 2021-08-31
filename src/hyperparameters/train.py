# Copyright (c) 2021 Alix Routhier-Lalonde. Licence included in root of package.

from src.model import GRUModel, LSTMModel, GRU_LSTM_Model
from torch.nn import MSELoss
from torch.optim import Adam


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
    epochs = 100
    patience = epochs
    learning_rate = 0.001
    validation_split = 0.1
    loss = MSELoss
    optimizer = Adam
