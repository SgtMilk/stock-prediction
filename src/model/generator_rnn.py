# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
Contains the Generator class, the generator model for the RNN GAN.
"""

import torch
from torch.nn import Module, GRU, Linear, ReLU


class GeneratorRNN(Module):
    """
    This model is a Generator for a RNN GAN.
    """

    def __init__(
        self,
        device,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        """
        Initializes the Generator model.
        :param device: the device to train on (cpu vs cuda)
        :param hidden_dim: the hidden dimension inside of the model
        :param num_layers: the number of gru layers in the model
        :param dropout: the dropout percentage in the gru layers
        """
        super(GeneratorRNN, self).__init__()

        self.gru = GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.linear = Linear(hidden_dim, 1)

        self.relu = ReLU()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, input_data, hidden):
        """
        Returns the model prediction
        :param input_data: the input to the prediction
        :param hidden: the hidden state of the model
        :return the predicted data, the hidden state
        """

        output, hidden = self.gru(input_data, hidden)
        output = self.relu(output[:, -1])
        output = self.linear(output)

        return output, hidden

    def init_hidden(self, batch_size):
        """
        Initiates the hidden layer of the GRU RNN model
        :param batch_size: the batch size of the input
        :return the initialized hidden state
        """

        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden
