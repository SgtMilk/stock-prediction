from torch.nn import Sequential, LSTM


def generate_model():
    """
    Will generate and return a Pytorch model
    :return: a pytorch model
    """
    return Sequential(LSTM())
