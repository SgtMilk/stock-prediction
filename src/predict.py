from src.hyperparameters import Train
from src.data import AggregateDataset
from src.model import Net
import numpy as np
import datetime
import os
import torch


def predict(codes, mode: int, overwrite: bool = False):
    """
    predict callback
    :param overwrite: overwrite flag
    :param codes: the codes to train to
    :param mode: Mode.daily, Mode.weekly, Mode.monthly
    :return: predicted data
    """
    # getting the right file path
    destination_folder = os.path.abspath('./src/model/models')
    code_string = ""
    for code in codes:
        code_string += f"{code}-"
    current_date = str(datetime.date.today())
    filepath = os.path.join(destination_folder, f"{code_string}{mode}-{current_date}.hdf5")

    # getting the data
    dataset = AggregateDataset(codes, mode=mode, y_flag=True)
    dataset.transform_to_torch()

    gpu = torch.cuda.is_available()

    # getting our model and net
    model = Train.model(dataset.x.shape[-1], Train.hidden_dim, Train.num_dim, Train.dropout, mode)

    if gpu:
        model.to('cuda')

    if not os.path.exists(filepath) or overwrite:
        net = Net(Train.optimizer(model.parameters(), lr=Train.learning_rate), Train.loss(reduction='mean'), model,
                  dataset)
        net.train(Train.epochs, dataset, Train.validation_split, Train.patience)
        model = net.model
    else:
        model.load_state_dict(torch.load(filepath))

    data = torch.from_numpy(np.array([dataset.x[0].cpu().numpy()])).float()
    if gpu:
        data = data.to(device='cuda')
    predicted_y_test = model(data)

    # re-transforming to numpy
    predicted_y_test = predicted_y_test.detach().cpu().numpy()

    return dataset.normalizer.inverse_transform(
        predicted_y_test)
