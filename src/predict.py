# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

from src.hyperparameters import Train
from src.data import Dataset
from src.model import Net
from src.utils import get_base_path
import numpy as np
import datetime
import os
import torch


def predict_stock(code, interval: int, num_days: int, overwrite: bool = False):
    """
    predict callback
    :param overwrite: overwrite flag
    :param code: the code to train to
    :param interval: Interval.daily, Interval.weekly, Interval.monthly
    :param num_days: the number of days to predict
    :return: predicted data
    """
    # getting the right file path
    destination_folder = os.path.abspath(
        os.path.join(get_base_path(), 'src/model/models'))
    filepath = os.path.join(
        destination_folder, f"model-{str(interval)}.hdf5")

    # getting the data
    dataset = Dataset(code, interval=interval, y_flag=True)
    dataset.transform_to_torch()

    gpu = torch.cuda.is_available()

    # getting our model and net
    model = Train.model(
        dataset.x.shape[-1], Train.hidden_dim, Train.num_dim, Train.dropout, 1)

    if gpu:
        model.to('cuda')

    if not os.path.exists(filepath) or overwrite:
        net = Net(Train.optimizer(model.parameters(), lr=Train.learning_rate), Train.loss(reduction='mean'), model,
                  dataset)
        net.train(Train.epochs, dataset,
                  Train.validation_split, Train.patience)
        model = net.model
    else:
        model.load_state_dict(torch.load(filepath))

    data = torch.from_numpy(np.array(dataset.x.detach().cpu().numpy())).float()
    if gpu:
        data = data.to(device='cuda')

    for i in range(num_days):
        prediction = model(data[-1].unsqueeze(0))
        temp = data[-1, 1:].detach().clone()
        concatenated_prediction = torch.cat((temp, prediction))
        formatted_prediction = concatenated_prediction.unsqueeze(0)
        torch.cat((data, formatted_prediction))

    # re-transforming to numpy
    print(data.detach().cpu().numpy()[-1, -(num_days + 1):].squeeze().shape)
    predicted = data.detach().cpu().numpy()[-1, -(num_days + 1):].squeeze()

    return inverse_scaling(predicted, dataset)

def inverse_scaling(scaled_data, dataset):
    first_unscaling = dataset.inverse_transform(scaled_data)
    print(f"${dataset.code}: ${dataset.y_unscaled.detach().cpu().numpy()[-1]}")
    scaling_factor = dataset.y_unscaled.detach().cpu().numpy()[-1]/first_unscaling[0]
    return first_unscaling * scaling_factor
