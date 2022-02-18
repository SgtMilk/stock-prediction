# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

import os
import numpy as np
import torch
from src.hyperparameters import GAN
from src.data import Dataset
from src.utils import get_base_path


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
        destination_folder, f"generator-{str(interval)}.hdf5")

    # getting the data
    dataset = Dataset(code, interval=interval, y_flag=True, pred_length=GAN.pred_length, look_back=GAN.look_back)
    dataset.transform_to_torch()

    # getting our model and net
    generator = GAN.generator(GAN.device, dataset.x.shape[-1], GAN.hidden_dim, GAN.num_dim, GAN.dropout, dataset.y.shape[-2], GAN.kernel_size)

    generator.to(device=GAN.device)

    if not os.path.exists(filepath) or overwrite:
        return None

    generator.load_state_dict(torch.load(filepath))

    data = torch.from_numpy(np.array(dataset.x.detach().cpu().numpy())).float().to(device=GAN.device)
    data = torch.reshape(data[-1], (1, dataset.look_back, 1))
    
    generator.eval()
    returned_data = None

    with torch.set_grad_enabled(False):
        if dataset.pred_length == 1:
            for _ in range(num_days):
                prediction = generator(data[:,-dataset.look_back:])
                prediction = torch.reshape(prediction, (prediction.shape[0], prediction.shape[1]))
                data = torch.cat((data, torch.reshape(prediction, (prediction.shape[0], 1, 1))), axis=1) 
                returned_data = data[:,-num_days:] 
        else:
            returned_data = generator(data).squeeze()

    # re-transforming to numpy
    predicted = returned_data.detach().cpu().numpy().squeeze()

    returned_data = inverse_scaling(predicted, dataset)[:num_days + 1]

    return returned_data

def inverse_scaling(scaled_data, dataset):
    """inverses the scaling from the dataset"""
    first_unscaling = dataset.inverse_transform(scaled_data.squeeze())
    scaling_factor = dataset.y_unscaled.detach().cpu().numpy()[-1, 0] - first_unscaling.squeeze()[0]
    return first_unscaling + scaling_factor
