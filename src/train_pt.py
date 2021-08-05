from data import Dataset, Mode, AggregateDataset, get_stock_symbols
from model_pt import Net, LSTMModel, GRUModel
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
import torch


def train_stock(codes, mode: int):
    """
    Trains one stock
    :param codes: array of stock codes
    :param mode: Mode.daily, Mode.weekly, Mode.monthly
    """
    dataset = AggregateDataset(codes, mode=mode, y_flag=True)
    dataset.transform_to_torch()
    model = LSTMModel(dataset.x.shape[-1], 128, 3, 0.2, mode)
    net = Net(Adam(model.parameters(), lr=0.001), MSELoss(reduction='mean'), model)
    net.train(5000, dataset.get_train(), 0.1)
    net.evaluate_training()
    net.evaluate(dataset)


if __name__ == "__main__":
    stock_symbols = ['AAPL']
    torch.manual_seed(1)
    train_stock(stock_symbols, Mode.monthly)
