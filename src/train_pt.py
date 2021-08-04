from data import Dataset, Mode
from model_pt import Net, LSTMModel, GRUModel
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
    dataset.transform_to_torch()
    model = GRUModel(6, 32, 3, 0.2, mode)
    net = Net(Adam(model.parameters(), lr=0.01), MSELoss(reduction='mean'), model)
    net.train(100, dataset.get_train(), 0.1)
    net.evaluate_training()
    net.evaluate(dataset)


if __name__ == "__main__":
    torch.manual_seed(1)
    train_stock("ARL", Mode.weekly)
