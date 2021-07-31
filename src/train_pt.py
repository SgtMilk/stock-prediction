from data import Dataset, Mode
from model_pt import Net, generate_model
from torch.nn import MSELoss
from torch.optim import Adam


def train_stock(code: str, mode: int):
    """
    Trains one stock
    :param code: the stock's code
    :param mode: Mode.daily, Mode.weekly, Mode.monthly
    """
    dataset = Dataset(code, mode=mode, y_flag=True)
    net = Net(Adam, MSELoss, generate_model)
    net.train(200, dataset, 0.1)
    net.evaluate(dataset)


if __name__ == "__main__":
    train_stock("ARL", Mode.daily)
