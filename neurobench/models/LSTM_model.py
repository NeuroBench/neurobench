"""
Class defines an LSTM model + a feedforward layer

Authors
~~~~~~~
Younes Bouhadjar
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM model

    Parameters:
        input_dim:   int
                     The number of expected features in the input x
        hidden_size: int 
                     The number of features in the hidden state of the LSTM
        n_layers:    int
                     The number of LSTM layers
        output_dim:  int 
                     The number of output features after passing through the fully connected layer


    """
    def __init__(self, input_dim=1, hidden_size=50, n_layers=2, output_dim=1, dtype=torch.float64):
        super().__init__()

        self.input_dim = input_dim

        self.rnn = nn.LSTM(input_dim, hidden_size, n_layers, batch_first=True).type(dtype)
        self.activation = nn.ReLU().type(dtype)
        self.fc1 = nn.Linear(hidden_size, output_dim).type(dtype)

    def forward(self, x):

        x, _ = self.rnn(x)
        x = self.activation(x)
        out = self.fc1(x)

        return out
