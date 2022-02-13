from torch.nn import Module, Sequential, Conv1d, BatchNorm1d, LeakyReLU, GRU, Sigmoid

class Discriminator(Module):
    """
    This model is a Discriminator for a GAN. It is the inverse of the Generator model which is located in the same folder.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, kernel_size):
        super(Discriminator, self).__init__()
        self.main = Sequential(
            # GRU(input_size=input_dim, hidden_size=hidden_dim * 8, num_layers=num_layers, dropout=dropout, batch_first=True),

            Conv1d(input_dim, hidden_dim * 4, kernel_size, bias=False),
            BatchNorm1d(hidden_dim * 4),
            LeakyReLU(True),

            Conv1d(hidden_dim * 4, hidden_dim * 2, kernel_size, bias=False),
            BatchNorm1d(hidden_dim * 2),
            LeakyReLU(True),

            Conv1d(hidden_dim * 2, hidden_dim, kernel_size, bias=False),
            BatchNorm1d(hidden_dim),
            LeakyReLU(True),

            Conv1d(hidden_dim, 1, kernel_size, bias=False),
            Sigmoid()
        )
    
    def forward(self, input):
        return self.main(input)