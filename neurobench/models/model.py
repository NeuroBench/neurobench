"""
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import spikegen
import math
from scipy import signal
import random

from neurobench import utils

class ANNModel(nn.Module):
    """
        A straightforward 3-layer fully-connected network for predicting x&y coordinate
        of the test subject.
        :param input_dim: input feature dimension
        :param layer1: Number of hidden neurons in the first layer
        :param layer2: Number of hidden neurons in the second layer
        :param output_dim: output feature dimension
        :param dropout_rate: Probability of dropout
        """

    def __init__(self, hyperparams, input_dim=96, layer1=32, layer2=48, layer3=48, output_dim=2, dropout_rate=0.5):
        super().__init__()
        self.timesteps = hyperparams['num_steps']
        self.fc1 = nn.Linear(input_dim, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm1 = nn.BatchNorm1d(layer1)
        self.batchnorm2 = nn.BatchNorm1d(layer2)
        self.hyperparams = hyperparams
        self.batch_size = None

    def forward(self, x):
        self.batch_size = x.shape[0]
        if not self.training:
            input_ = torch.clone(x)

        x = self.activation(self.fc1(x.view(self.batch_size, -1)))
        x = self.batchnorm1(x)

        if not self.training:
            fc1_output = torch.clone(x)

        x = self.activation(self.dropout(self.fc2(x)))
        x = self.batchnorm2(x)

        if not self.training:
            fc2_output = torch.clone(x)

        x = self.fc3(x)

        if self.training:
            return x    # Training Mode only need to return the output

        # Inference Mode returns the followin: (final_output, input, fc1_output, fc2_output)
        return x, input_, fc1_output, fc2_output

class SNNModel(nn.Module):

    def __init__(self, beta=0.95, mem_threshold=1.0, spike_grad1=surrogate.fast_sigmoid(slope=20),
                 input_dim=96, layer1=32, layer2=48, output_dim=2, dropout_rate=0.5, num_step=15, data_mode="2D"):
        super().__init__()

        self.num_step = num_step
        self.data_mode = data_mode
        self.fc1 = nn.Linear(input_dim, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, output_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad1, threshold=mem_threshold, learn_beta=True,
                              learn_threshold=True)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad1, threshold=mem_threshold, learn_beta=True,
                              learn_threshold=True)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad1, threshold=mem_threshold, learn_beta=True,
                              learn_threshold=True, reset_mechanism="none")
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm1 = nn.BatchNorm1d(layer1)
        self.batchnorm2 = nn.BatchNorm1d(layer2)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_rec = []
        mem_rec = []

        for step in range(self.num_step):
            if self.data_mode == "2D":
                cur1 = self.batchnorm1(self.dropout(self.fc1(x)))
                spk1, mem1 = self.lif1(cur1, mem1)

            elif self.data_mode == "3D":
                cur1 = self.batchnorm1(self.dropout(self.fc1(x[step])))
                spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.batchnorm2(self.fc2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk_rec.append(spk3)
            mem_rec.append(mem3)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

class BesselFilter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        data = input_.detach().cpu().numpy()
        b, a = signal.bessel(4, 0.05, btype="low", analog=False) # order=4, cutoff=0.05 
        y = signal.filtfilt(b, a, data)

        return torch.as_tensor(y.copy(), dtype=input_.dtype, device=input_.device)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        
        return grad_input

def bessel_filter():
    def inner(x):
        return BesselFilter.apply(x)

    return inner

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim, batch_size, timesteps, bottle_size=32, droprate=0.5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.bottle_size = bottle_size

        self.norm_layer = nn.LayerNorm([self.timesteps, self.input_size])
        
        self.conv = nn.Conv1d(self.input_size, self.hidden_size, 1)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, 1, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(droprate)
        self.fc1 = nn.Linear(self.hidden_size, self.output_dim) 
        self.bottleneck = nn.Linear(self.input_size, self.bottle_size)

        self.filter = bessel_filter()

    def forward(self, x, h_t, c_t):
        # Expected Input Feature Dimension: Dimension * Timesteps * Input_Features
        x = self.norm_layer(x)

        x = torch.permute(x, (0, 2, 1))
        x = self.conv(x)
        x = torch.permute(x, (0, 2, 1))

        output, (h, c) = self.lstm(x, (h_t, c_t))
        output = self.dropout(output)

        x = self.fc1(output)

        x = x[:, -1, :].t()
        filtered_result = self.filter(x)
        filtered_result = filtered_result.t()

        return filtered_result, (h, c)
