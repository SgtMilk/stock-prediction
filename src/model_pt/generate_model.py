from torch.nn import LSTM, Linear, Module, ReLU
from src.data import Dataset
import torch
from torch.autograd import Variable


class LSTMModel(Module):
    def __init__(self, dataset: Dataset):
        super(LSTMModel, self).__init__()
        self.input_shape = dataset.x.shape
        self.num_layers = 3

        self.lstm = LSTM(
            input_size=self.input_shape[-1],
            hidden_size=self.input_shape[-1],
            num_layers=self.num_layers,
            dropout=0.2
        )
        self.fc = Linear(self.input_shape[-1], dataset.mode)
        self.relu = ReLU

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))   # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))   # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))     # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)    # first Dense
        out = self.relu(out)    # relu
        out = self.fc(out)  # Final Output
        return out
