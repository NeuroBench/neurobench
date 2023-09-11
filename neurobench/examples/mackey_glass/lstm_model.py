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

    Parameters: #TODO
        input_dim:   int
                     The number of expected features in the input x
        hidden_size: int 
                     The number of features in the hidden state of the LSTM
        n_layers:    int
                     The number of LSTM layers
        output_dim:  int 
                     The number of output features after passing through the fully connected layer


    """
    def __init__(
        self, 
        input_dim: int = 1, 
        hidden_size: int = 50, 
        n_layers: int = 2, 
        output_dim: int = 1, 
        dropout_rate: float = 0.5, 
        mode: str = "autonomous",
        dtype=torch.float64
        ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.mode = mode

        assert self.mode in ["autonomous", "single_step"]
        self.prior_prediction = None
    
        self.rnn = nn.LSTMCell(self.input_dim, 
                               self.hidden_size).type(dtype)
        #self.drop = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU().type(dtype)
        self.fc = nn.Linear(hidden_size, output_dim).type(dtype)

    def single_forward(self, sample): 
        x, _ = self.rnn(sample)
        x = self.activation(x)
        #x = self.drop(x)
        out = self.fc(x)

        return out

    def forward(self, x):
        predictions = []
        for sample in x:
            if self.mode == 'autonomous' and self.prior_prediction is not None:
                sample = self.prior_prediction
            prediction = self.single_forward(sample)
            predictions.append(prediction)
            self.prior_prediction = prediction

        # reset so that next batch will not have prior prediction
        self.prior_prediction = None

        return torch.stack(predictions).squeeze(-1)
