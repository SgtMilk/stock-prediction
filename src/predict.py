# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
Contains the predict_stock function. Will predict stock data for the backend.
"""

import os
import numpy as np
import torch
from src.hyperparameters import GAN
from src.data import Dataset
from src.utils import get_base_path


def predict_stock(code, pred_length: int):
    """
    predict function for the backend.
    According to a stock code, it will download the latest data on that stock and return
    a prediction.
    This function returns None if there are no trained models available for the inputted interval.
    :param code: the code of the stock to predict
    :param pred_length: the number of days to predict
    :return: predicted data or None
    """
    # getting the right file path
    destination_folder = os.path.abspath(os.path.join(get_base_path(), "src/model/models"))
    filepath = os.path.join(destination_folder, "model.hdf5")

    # getting the data
    dataset = Dataset(
        GAN.device,
        code,
        y_flag=True,
    )
    dataset.transform_to_torch()

    # getting our model and net
    model = GAN.model(GAN.device, GAN.hidden_dim, GAN.num_dim, GAN.dropout)

    model.to(device=GAN.device)

    if not os.path.exists(filepath):
        return None

    model.load_state_dict(torch.load(filepath))

    data = (
        torch.from_numpy(np.array(dataset.x_data.detach().cpu().numpy()))
        .float()
        .to(device=GAN.device)
    )[-GAN.look_back :]

    # getting the predicted prices
    model.eval()

    prediction = [data[-1]]

    with torch.set_grad_enabled(False):
        hidden = model.init_hidden(1)
        for price in data[:-1]:
            _, hidden = model(price, hidden)
        for _ in range(pred_length):
            output, hidden = model(prediction[-1], hidden)
            prediction.append(output.squeeze())
    return [x.item() for x in prediction]


def inverse_scaling(scaled_data, dataset):
    """inverses the scaling from the dataset"""
    first_unscaling = dataset.inverse_transform(
        scaled_data.reshape((1, scaled_data.shape[0]))
    ).squeeze()
    scaling_factor = dataset.y_unscaled[-1].item() - first_unscaling[0]
    return first_unscaling + scaling_factor
