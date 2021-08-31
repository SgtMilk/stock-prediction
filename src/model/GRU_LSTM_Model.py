# Copyright (c) 2021 Alix Routhier-Lalonde. Licence included in root of package.

from torch.nn import Linear, Module, GRU, LSTM
import torch


class GRU_LSTM_Model(Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim):
        """
        initiates the model
        :param input_dim: The dimension of the input tensor
        :param hidden_dim: The dimension inside of of the model
        :param num_layers: The number of GRU layers
        :param dropout: The dropout rate at every layer
        :param output_dim: The dimension of the output tensor
        """
        super(GRU_LSTM_Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim

        self.gru = GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )

        self.lstm = LSTM(
            input_size=self.input_dim,
            hidden_size=self.input_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )

        self.fc = Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        """
        the forward method of the model, will predict outputs
        :param x: input tensor
        :return: the predicted output
        """
        gpu = torch.cuda.is_available()
        h0 = torch.randn(self.num_layers, x.size(
            0), self.input_dim).requires_grad_()
        c0 = torch.randn(self.num_layers, x.size(
            0), self.input_dim).requires_grad_()
        h1 = torch.randn(self.num_layers, x.size(
            0), self.hidden_dim).requires_grad_()
        if gpu:
            h0 = h0.to(device='cuda')
            c0 = c0.to(device='cuda')
            h1 = h1.to(device='cuda')
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out, (hn) = self.gru(out, (h1.detach()))
        out = self.fc(out[:, -1, :])
        return out