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


def predict_stock(code, interval: int, pred_length: int):
    """
    predict function for the backend.
    According to a stock code, it will download the latest data on that stock and return
    a prediction.
    This function returns None if there are no trained models available for the inputted interval.
    :param code: the code of the stock to predict
    :param interval: Interval.daily, Interval.weekly, Interval.monthly
    :param pred_length: the number of days to predict
    :return: predicted data or None
    """
    # getting the right file path
    destination_folder = os.path.abspath(os.path.join(get_base_path(), "src/model/models"))
    filepath = os.path.join(destination_folder, f"generator-{str(interval)}.hdf5")

    # getting the data
    dataset = Dataset(
        GAN.device,
        code,
        interval=interval,
        y_flag=True,
        pred_length=GAN.pred_length,
        look_back=GAN.look_back,
    )
    dataset.transform_to_torch()

    # getting our model and net
    generator = GAN.generator(
        GAN.device,
        dataset.x_data.shape[-2],
        GAN.hidden_dim,
        dataset.y_data.shape[-2],
        GAN.num_dim,
        GAN.dropout,
        GAN.kernel_size,
    )

    generator.to(device=GAN.device)

    if not os.path.exists(filepath):
        return None

    generator.load_state_dict(torch.load(filepath))

    data = (
        torch.from_numpy(np.array(dataset.x_data.detach().cpu().numpy()))
        .float()
        .to(device=GAN.device)
    )
    data = torch.reshape(data[-1], (1, dataset.look_back, 1))

    # getting the predicted prices
    generator.eval()
    returned_data = None

    with torch.set_grad_enabled(False):
        if dataset.pred_length == 1:
            for _ in range(pred_length):
                prediction = generator(data[:, -dataset.look_back :])
                prediction = torch.reshape(prediction, (prediction.shape[0], prediction.shape[1]))
                data = torch.cat(
                    (data, torch.reshape(prediction, (prediction.shape[0], 1, 1))), axis=1
                )
                returned_data = data[:, -pred_length:]
        else:
            returned_data = generator(data).squeeze()

    # re-transforming to numpy
    predicted = returned_data.detach().cpu().numpy().squeeze()

    returned_data = inverse_scaling(predicted, dataset)[: pred_length + 1]

    return returned_data


def inverse_scaling(scaled_data, dataset):
    """inverses the scaling from the dataset"""
    first_unscaling = dataset.inverse_transform(
        scaled_data.reshape((1, scaled_data.shape[0]))
    ).squeeze()
    scaling_factor = dataset.y_unscaled[-1, -1].item() - first_unscaling[0]
    return first_unscaling + scaling_factor
