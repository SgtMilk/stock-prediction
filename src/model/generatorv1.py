# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
Contains the Generator class, the generator model for the GAN.
"""

import torch
from torch.nn import Module, BatchNorm1d, LeakyReLU, GRU, Linear, Conv1d, Tanh


class GeneratorV1(Module):
    """
    This model is a Generator for a GAN. I added temporality with a GRU layer and then convolutional
    transpose layers.
    """

    def __init__(
        self,
        device,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        kernel_size: int,
    ):
        """
        Initializes the Generator model.
        :param device: the device to train on (cpu vs cuda)
        :param input_dim: the number of input features
        :param hidden_dim: the hidden dimension inside of the model
        :param ouput_dim: the number of ouput features to predict
        :param num_layers: the number of gru layers in the model
        :param dropout: the dropout percentage in the gru layers
        :param kernel_size: the kernel size in the convolution layers
        """
        super(GeneratorV1, self).__init__()

        self.output_dim = output_dim

        self.conv = Conv1d(input_dim, hidden_dim * 2, kernel_size, bias=False)
        self.batch = BatchNorm1d(hidden_dim * 2)
        self.relu = LeakyReLU(True)

        self.gru = GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.linear = Linear(hidden_dim, 1)

        self.tanh = Tanh()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.input_dim = input_dim

    def forward(self, input_data):
        """
        Predicts from model output_dim times to have an output of the right format.
        :param input_data: the input to the prediction
        :return the predicted data
        """
        for _ in range(self.output_dim):
            prediction = self.forward_helper(input_data[:, -self.input_dim :])
            input_data = torch.cat(
                (input_data, torch.reshape(prediction, (prediction.shape[0], 1, 1))), axis=1
            )
        returned_data = input_data[:, -self.output_dim :]
        return returned_data

    def forward_helper(self, input_data):
        """
        Returns the model prediction
        :param input_data: the input to the prediction
        :return the predicted data
        """

        output = self.conv(input_data)
        output = self.batch(output)
        output = self.relu(output)

        noise = (
            torch.randn(self.num_layers, output.size(0), self.hidden_dim)
            .requires_grad_()
            .to(device=self.device)
        )
        output, (_) = self.gru(output.permute(0, 2, 1), noise)
        output = self.linear(output[:, -1])
        output = self.tanh(output)

        return output
