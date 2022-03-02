# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
Contains the Generator class, the generator model for the GAN.
"""

import torch
from torch.nn import Module, BatchNorm1d, GRU, Linear, Sequential, ConvTranspose1d, ReLU, Tanh


class GeneratorV3(Module):
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
        super(GeneratorV3, self).__init__()

        self.output_dim = output_dim

        self.gru = GRU(
            input_size=input_dim,
            hidden_size=hidden_dim * 4,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.conv = Sequential(
            ConvTranspose1d(hidden_dim * 4 + input_dim, hidden_dim * 2, kernel_size, bias=False),
            BatchNorm1d(hidden_dim * 2),
            ReLU(True),
            ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size, bias=False),
            BatchNorm1d(hidden_dim),
            ReLU(True),
            ConvTranspose1d(hidden_dim, output_dim, kernel_size, bias=False),
            Tanh(),
        )

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.input_dim = input_dim

    def forward(self, input_data):
        """
        Returns the model prediction
        :param input_data: the input to the prediction
        :return the predicted data
        """

        noise_gru = (
            torch.randn(self.num_layers, input_data.size(0), self.hidden_dim * 4)
            .requires_grad_()
            .to(device=self.device)
        )
        output, (_) = self.gru(input_data.permute(0, 2, 1), noise_gru)

        noise_input = torch.randn(input_data.shape).requires_grad_().to(device=self.device)

        output = torch.cat((noise_input, output[:, -1:, :].permute(0, 2, 1)), 1)

        output = self.conv(output)

        return output
