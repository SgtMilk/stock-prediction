# Copyright (c) 2021 Alix Routhier-Lalonde. Licence included in root of package.

from src.hyperparameters import Train
from src.data import Dataset
from src.model import Net
from src.utils import get_base_path
import numpy as np
import datetime
import os
import torch


def predict_stock(code, mode: int, overwrite: bool = False):
    """
    predict callback
    :param overwrite: overwrite flag
    :param code: the code to train to
    :param mode: Mode.daily, Mode.weekly, Mode.monthly
    :return: predicted data
    """
    # getting the right file path
    destination_folder = os.path.abspath(
        os.path.join(get_base_path(), 'src/model/models'))
    filepath = os.path.join(
        destination_folder, f"model-{str(mode)}.hdf5")

    # getting the data
    dataset = Dataset(code, mode=mode, y_flag=True)
    dataset.transform_to_torch()

    gpu = torch.cuda.is_available()

    # getting our model and net
    model = Train.model(
        dataset.x.shape[-1], Train.hidden_dim, Train.num_dim, Train.dropout, mode)

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

    data = torch.from_numpy(np.array(dataset.prediction_data)).float()
    if gpu:
        data = data.to(device='cuda')
    predicted = model(data)

    # re-transforming to numpy
    predicted = predicted.detach().cpu().numpy()

    return dataset.normalizer.inverse_transform(predicted)
