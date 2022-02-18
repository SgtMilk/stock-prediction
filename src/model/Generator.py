from torch.nn import Module, Sequential, BatchNorm1d, LeakyReLU, GRU, Linear, Flatten, Conv1d
import torch

class Generator(Module):
    """
    This model is a Generator for a GAN. I added temporality with a GRU layer and then convolutional transpose layers.
    """
    def __init__(self, device, input_dim, hidden_dim, num_layers, dropout, output_dim, kernel_size):
        super(Generator, self).__init__()

        self.output_dim = output_dim

        self.conv = Conv1d(input_dim, hidden_dim * 2, kernel_size, bias=False)
        self.batch = BatchNorm1d(hidden_dim * 2)
        self.relu = LeakyReLU(True)

        self.gru = GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True)

        self.linear = Linear(hidden_dim, 1)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.input_dim = input_dim
    
    def forward(self, input_data):
        """
        predicts from model
        """
        for _ in range(self.output_dim):
            prediction = self.forward_helper(input_data[:,-self.input_dim:])
            input_data = torch.cat((input_data, torch.reshape(prediction, (prediction.shape[0], 1, 1))), axis=1)  
        returned_data = input_data[:,-self.output_dim:]
        return returned_data

    def forward_helper(self, input_data):
        """
        Returns the model prediction
        """

        t = self.conv(input_data)
        t = self.batch(t)
        t = self.relu(t)

        h0 = torch.randn(self.num_layers, t.size(0), self.hidden_dim).requires_grad_().to(device=self.device)
        t, (h_n) = self.gru(t.permute(0,2,1), h0)
        t = self.linear(t[:, -1])

        return t
