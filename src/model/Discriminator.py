from torch.nn import Module, Sequential, Conv1d, BatchNorm1d, LeakyReLU, GRU, Sigmoid, Linear
import torch

class Discriminator(Module):
    """
    This model is a Discriminator for a GAN. It is the inverse of the Generator model which is located in the same folder.
    """
    def __init__(self, device, input_dim, hidden_dim, num_layers, dropout, kernel_size):
        super(Discriminator, self).__init__()

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
        self.sigmoid = Sigmoid()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = device
    
    def forward(self, input_data):
        """
        Returns the model prediction
        """

        t = self.conv(input_data)
        t = self.batch(t)
        t = self.relu(t)

        h0 = torch.randn(self.num_layers, t.size(0), self.hidden_dim).requires_grad_().to(device=self.device)
        t, (_) = self.gru(t.permute(0,2,1), h0)
        t = self.linear(t[:, -1])
        t = self.sigmoid(t)

        return t