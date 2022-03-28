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
    filepath = os.path.join(destination_folder, "generator.hdf5")

    # getting the data
    dataset = Dataset(
        GAN.device,
        code,
        y_flag=True,
    )
    dataset.transform_to_torch()

    # getting our model and net
    generator = GAN.generator(
        GAN.device, GAN.hidden_dim, GAN.num_dim, GAN.dropout, noise_proportion=GAN.noise_proportion
    )

    generator.to(device=GAN.device)

    if not os.path.exists(filepath):
        return None

    generator.load_state_dict(torch.load(filepath))

    data = (
        torch.from_numpy(np.array(dataset.x_data.detach().cpu().numpy()))
        .float()
        .to(device=GAN.device)
    )[-GAN.look_back :]

    # getting the predicted prices
    generator.eval()
    returned_data = None

    with torch.set_grad_enabled(False):
        hidden = generator.init_hidden(1)
        for price in data:
            _, hidden = generator(expand_dims(price), hidden)
        for _ in range(pred_length):
            output, hidden = generator(expand_dims(data[-1]), hidden)
            torch.cat((data, torch.tensor([output.squeeze()]).to(GAN.device)))

    # re-transforming to numpy
    predicted = data[-(pred_length + 1) :].detach().cpu().numpy().squeeze()

    returned_data = inverse_scaling(predicted, dataset)

    return returned_data


def inverse_scaling(scaled_data, dataset):
    """inverses the scaling from the dataset"""
    first_unscaling = dataset.inverse_transform(
        scaled_data.reshape((1, scaled_data.shape[0]))
    ).squeeze()
    scaling_factor = dataset.y_unscaled[-1].item() - first_unscaling[0]
    return first_unscaling + scaling_factor


def expand_dims(value):
    """expands the dims for the generator"""
    return torch.unsqueeze(torch.unsqueeze(torch.tensor([value]), axis=1), axis=2).to(GAN.device)
