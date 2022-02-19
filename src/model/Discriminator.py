# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
Contains the Discriminator class, the discriminator model for the GAN.
"""

from torch.nn import Module, Conv1d, BatchNorm1d, LeakyReLU, GRU, Sigmoid, Linear
import torch


class Discriminator(Module):
    """
    This model is a Discriminator for a GAN. It is the inverse of the Generator model which is
    located in the same folder.
    """

    def __init__(self, device, input_dim, hidden_dim, num_layers, dropout, kernel_size):
        """
        Initializes the Discriminator model.
        :param device: the device to train on (cpu vs cuda)
        :param input_dim: the number of input features
        :param hidden_dim: the hidden dimension inside of the model
        :param num_layers: the number of gru layers in the model
        :param dropout: the dropout percentage in the gru layers
        :param kernel_size: the kernel size in the convolution layers
        """
        super(Discriminator, self).__init__()

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
        self.sigmoid = Sigmoid()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, input_data):
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
        output = self.sigmoid(output)

        return output
