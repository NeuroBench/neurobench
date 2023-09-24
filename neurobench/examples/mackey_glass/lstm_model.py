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
        mode: str = 'autonomous',
        dtype = torch.float64,
        device = torch.device("cpu")
        ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.mode = mode
        self.dtype = dtype
        self.device = device

        assert self.mode in ['autonomous', 'single_step']
        #assert n_layers == 1, 'multi-layer LSTM is not supported yet'
        self.prior_prediction = None
 
        # LSTM model
        self.rnns = nn.ModuleList()
        self.rnns.append(nn.LSTMCell(
            self.input_dim, self.hidden_size).type(dtype)
        )
        self.rnns.extend(
            [nn.LSTMCell(self.hidden_size, self.hidden_size).type(dtype)
             for _ in range(1, self.n_layers)])

        self.activation = nn.ReLU().type(dtype)
        self.fc = nn.Linear(hidden_size, output_dim).type(dtype)
    
        # stores lookback window time steps
        self.register_buffer('inp', torch.zeros(1, self.input_dim).to(dtype))

    def single_forward(self, sample): 

        x, _ = self.rnns[0](sample)
        for i in range(1, self.n_layers):
            x, _ = self.rnns[i](x)
        x = self.activation(x)
        out = self.fc(x)

        return out

    def forward(self, batch):

        predictions = []
        for i, sample in enumerate(batch):

            if self.mode == 'autonomous' and self.prior_prediction is not None:
                # push new element
                self.inp = torch.cat((self.inp, self.prior_prediction), dim=-1)
                # pop oldest element
                self.inp = self.inp[:, 1:]
                # update register
                inp = self.inp.clone()
            else:
                self.inp = torch.cat((self.inp, sample), dim=-1)
                self.inp = self.inp[:, 1:]
                inp = self.inp.clone()
       
            prediction = self.single_forward(inp)
            predictions.append(prediction)
            self.prior_prediction = prediction

        # reset so that next batch will not have prior prediction
        self.prior_prediction = None
        self.inp[:] = 0.

        return torch.stack(predictions).squeeze(-1)
