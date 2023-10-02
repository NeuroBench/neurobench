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
        self.layer_norm = nn.LayerNorm(self.input_dim).type(dtype) 

        # Create register buffers to store lookback window as well as for the LSTM states
        # This allows assessment of model memory footprint

        # Stores time steps of lookback window
        self.register_buffer('inp', torch.zeros(1, self.input_dim).type(dtype))

        # Stores hidden hi states
        for i in range(self.n_layers):
            self.register_buffer(f'h{i}', torch.zeros(1, self.hidden_size).type(self.dtype))

        # Stores ci states
        for i in range(self.n_layers):
            self.register_buffer(f'c{i}', torch.zeros(1, self.hidden_size).type(self.dtype))

        #TODO h and c are defined in forward pass to ensure they are in the correct device,
        # see if there is a cleaner way
        self.h = []
        self.c = []
        
    def single_forward(self, sample): 

        sample = self.layer_norm(sample)
        self.h[0], self.c[0] = self.rnns[0](sample, (self.h[0], self.c[0]))
        for i in range(1, self.n_layers):
            self.h[i], self.c[i] = self.rnns[i](self.h[i-1], (self.h[i], self.c[i]))
        x = self.activation(self.h[-1])
        out = self.fc(x)

        return out

    def forward(self, batch):

        # Reset register buffers 
        for i in range(self.n_layers):
            getattr(self, f'h{i}')[:] = 0.
        for i in range(self.n_layers):
            getattr(self, f'c{i}')[:] = 0.
        self.inp[:] = 0.

        self.h = [getattr(self, f'h{i}') for i in range(self.n_layers)]
        self.c = [getattr(self, f'c{i}') for i in range(self.n_layers)]

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

        return torch.stack(predictions).squeeze(-1)
