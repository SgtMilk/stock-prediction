from torch.nn import Module, Sequential, ConvTranspose1d, BatchNorm1d, ReLU, GRU, Tanh

class Generator(Module):
    """
    This model is a Generator for a GAN. I added temporality with a GRU layer and then convolutional transpose layers.
    
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim, kernel_size):
        super(Generator, self).__init__()
        self.main = Sequential(
            GRU(input_size=input_dim, hidden_size=hidden_dim * 4, num_layers=num_layers, dropout=dropout, batch_first=True),

            ConvTranspose1d(input_dim, hidden_dim * 4, kernel_size, bias=False),
            BatchNorm1d(hidden_dim * 4),
            ReLU(True),

            ConvTranspose1d(hidden_dim * 4, hidden_dim * 2, kernel_size, bias=False),
            BatchNorm1d(hidden_dim * 2),
            ReLU(True),

            ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size, bias=False),
            BatchNorm1d(hidden_dim),
            ReLU(True),

            ConvTranspose1d(hidden_dim, output_dim, kernel_size, bias=False),
            Tanh()
        )
    
    def forward(self, input):
        return self.main(input)

