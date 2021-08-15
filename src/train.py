from src.data import Mode, AggregateDataset, get_stock_symbols
from src.hyperparameters import Train
from src.model import Net
import torch


def train_stock(codes, mode: int):
    """
    Trains one stock
    :param codes: array of stock codes
    :param mode: Mode.daily, Mode.weekly, Mode.monthly
    """
    # getting the data
    dataset = AggregateDataset(codes, mode=mode, y_flag=True)
    dataset.transform_to_torch()

    # getting our model and net
    model = Train.model(dataset.x.shape[-1], Train.hidden_dim, Train.num_dim, Train.dropout, mode)
    net = Net(Train.optimizer(model.parameters(), lr=Train.learning_rate), Train.loss(reduction='mean'), model, dataset)

    # training and evaluating our model
    net.train(Train.epochs, dataset, Train.validation_split, Train.patience)
    net.evaluate_training()
    net.evaluate(dataset)
    return net.model


if __name__ == "__main__":
    stock_symbols = ['AAPL']
    torch.manual_seed(1)
    train_stock(stock_symbols, Mode.monthly)
