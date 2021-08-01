from data import Dataset, Mode
from model_pt import Net, generate_model
from torch.nn import MSELoss
from torch.optim import Adam
import torch


def train_stock(code: str, mode: int):
    """
    Trains one stock
    :param code: the stock's code
    :param mode: Mode.daily, Mode.weekly, Mode.monthly
    """
    dataset = Dataset(code, mode=mode, y_flag=True)
    net = Net(Adam, MSELoss, generate_model(mode))
    net.train(200, dataset.get_train(), 0.1)
    net.evaluate(dataset.get_test())


if __name__ == "__main__":
    torch.manual_seed(1)
    train_stock("ARL", Mode.daily)
