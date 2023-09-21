import snntorch as snn
from snntorch import utils

import torch
import torch.nn as nn


class CSNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(6, 200, 2, padding=4)
        self.max1 = nn.MaxPool2d(2)
        self.leaky1 = snn.Leaky(beta=0.4, init_hidden=True, threshold=0.002)
        self.conv2 = nn.Conv1d(100, 256, kernel_size=1)
        self.max2 = nn.MaxPool2d(2)
        self.leaky2 = snn.Leaky(beta=0.3, init_hidden=True, threshold=0.001)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(256, 7)
        self.leaky3 = snn.Leaky(beta=0.5, output=True, init_hidden=True, threshold=0.001)
        self.num_steps = 40

    def forward(self, input):
        x = self.conv1(input.reshape(input.shape[0], input.shape[1], 1))
        x = self.max1(x)
        x = self.leaky1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.leaky2(x)
        x = self.flatten(x)
        x = self.linear(x)
        spk_out, mem_out = self.leaky3(x)
        return spk_out, mem_out

    def single_forward(self, input):
        # Initialize hidden states and outputs at t=0
        mem_rec = []
        spk_rec = []

        self.leaky1.init_leaky()
        self.leaky2.init_leaky()
        self.leaky3.init_leaky()
        utils.reset(self)
        for step in range(self.num_steps):
            new_input = input[:, step, :]
            x = self.conv1(new_input.reshape(new_input.shape[0], new_input.shape[1], 1))
            x = self.max1(x)
            x = self.leaky1(x)
            x = self.conv2(x)
            x = self.max2(x)
            x = self.leaky2(x)
            x = self.flatten(x)
            x = self.linear(x)
            spk_out, mem_out = self.leaky3(x)

            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec), torch.stack(mem_rec)