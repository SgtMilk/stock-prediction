from torch.nn import Linear, Module, GRU
import torch


class GRUModel(Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim):
        """
        initiates the model
        :param input_dim: The dimension of the input tensor
        :param hidden_dim: The dimension inside of of the model
        :param num_layers: The number of GRU layers
        :param dropout: The dropout rate at every layer
        :param output_dim: The dimension of the output tensor
        """
        super(GRUModel, self).__init__()
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

        self.fc = Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        """
        the forward method of the model, will predict outputs
        :param x: input tensor
        :return: the predicted output
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out


