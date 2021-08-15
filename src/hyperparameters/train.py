from src.model import GRUModel, LSTMModel
from torch.nn import MSELoss
from torch.optim import Adam


class Train:
    """
    This class contains all the hyperparameters for training the model
    """
    # model parameters
    model = LSTMModel
    hidden_dim = 128
    num_dim = 3
    dropout = 0.2

    # training parameters
    epochs = 10
    patience = epochs
    learning_rate = 0.001
    validation_split = 0.1
    loss = MSELoss
    optimizer = Adam
