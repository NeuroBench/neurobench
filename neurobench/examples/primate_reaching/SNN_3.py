import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate

import numpy as np

class SNNModel3(nn.Module):
    def __init__(self, input_dim, layer1=32, layer2=48, output_dim=2,
                 batch_size=256, bin_window=0.2, num_steps=7, drop_rate=0.5,
                 beta=0.5, mem_thresh=0.5, spike_grad=surrogate.atan(alpha=2)):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = layer1
        self.layer2 = layer2
        self.drop_rate = drop_rate

        self.batch_size = batch_size
        self.bin_window_time = bin_window
        self.num_steps = num_steps
        self.sampling_rate = 0.004
        self.bin_window_size = int(self.bin_window_time / self.sampling_rate)
        self.step_size = self.bin_window_size // self.num_steps

        self.fc1 = nn.Linear(self.input_dim, self.layer1)
        self.fc2 = nn.Linear(self.layer1, self.layer2)
        self.fc3 = nn.Linear(self.layer2, self.output_dim)
        self.dropout = nn.Dropout(self.drop_rate)
        self.norm_layer = nn.LayerNorm([self.num_steps, self.input_dim])

        self.beta = beta
        self.mem_thresh = mem_thresh
        self.spike_grad = spike_grad
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, threshold=self.mem_thresh, 
                              learn_beta=False, learn_threshold=False, init_hidden=True)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, threshold=self.mem_thresh, 
                              learn_beta=False, learn_threshold=False, init_hidden=True)
        self.lif3 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, threshold=self.mem_thresh, 
                              learn_beta=False, learn_threshold=False, init_hidden=True, reset_mechanism="none")
        
        self.v_x = torch.nn.Parameter(torch.normal(0, 1, size=(1,), requires_grad=True))
        self.v_y = torch.nn.Parameter(torch.normal(0, 1, size=(1,), requires_grad=True))

        self.register_buffer("data_buffer", torch.zeros(1, input_dim).type(torch.float32), persistent=False)

    def reset_mem(self):
        self.lif1.reset_hidden()
        self.lif2.reset_hidden()
        self.lif3.reset_hidden()

    def single_forward(self, x):
        x = self.norm_layer(x)
        self.reset_mem()
        for step in range(self.num_steps):
            cur1 = self.dropout(self.fc1(x[step, :]))
            spk1 = self.lif1(cur1)

            cur2 = self.fc2(spk1)
            spk2 = self.lif2(cur2)

            cur3 = self.fc3(spk2)
            spk3 = self.lif3(cur3)

        return self.lif3.mem.clone()

    def forward(self, x):
        predictions = []

        seq_length = x.shape[0]
        for seq in range(seq_length):
            current_seq = x[seq, :, :]
            self.data_buffer = torch.cat((self.data_buffer, current_seq), dim=0)

            if self.data_buffer.shape[0] > self.bin_window_size:
                self.data_buffer = self.data_buffer[1:, :]

            # Accumulate
            spikes = self.data_buffer.clone()
            
            acc_spikes = torch.zeros((self.num_steps, self.input_dim))
            for i in range(self.num_steps):
                temp = torch.sum(spikes[self.step_size*i:self.step_size*i+(self.step_size), :], dim=0)
                acc_spikes[i, :] = temp

            pred = self.single_forward(acc_spikes)
            U_x = self.v_x*pred[0]
            U_y = self.v_y*pred[1]
            out = torch.stack((U_x, U_y), 0).permute(1, 0)
            predictions.append(out)

        predictions = torch.stack(predictions).squeeze(dim=1)

        return predictions