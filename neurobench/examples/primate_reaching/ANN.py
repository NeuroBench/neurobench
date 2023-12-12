import torch
import torch.nn as nn

import numpy as np

## Define model ##
# The model defined here is a vanilla Fully Connected Network
class ANNModel2D(nn.Module):
    def __init__(self, input_dim, layer1=32, layer2=48, output_dim=2,
                 bin_window=0.2, drop_rate=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = layer1
        self.layer2 = layer2
        self.drop_rate = drop_rate

        self.bin_window_time = bin_window
        self.sampling_rate = 0.004
        self.bin_window_size = int(self.bin_window_time / self.sampling_rate)

        self.fc1 = nn.Linear(self.input_dim, self.layer1)
        self.fc2 = nn.Linear(self.layer1, self.layer2)
        self.fc3 = nn.Linear(self.layer2, self.output_dim)
        self.dropout = nn.Dropout(self.drop_rate)
        self.batchnorm1 = nn.BatchNorm1d(self.layer1)
        self.batchnorm2 = nn.BatchNorm1d(self.layer2)
        self.activation = nn.ReLU()

        self.register_buffer("data_buffer", torch.zeros(1, input_dim).type(torch.float32), persistent=False)

    def single_forward(self, x):
        x = self.activation(self.fc1(x.view(1, -1)))
        x = self.batchnorm1(x)
        x = self.activation(self.dropout(self.fc2(x)))
        x = self.batchnorm2(x)
        x = self.fc3(x)

        return x

    def forward(self, x):
        predictions = []

        seq_length = x.shape[0]
        for seq in range(seq_length):
            current_seq = x[seq, :, :]
            self.data_buffer = torch.cat((self.data_buffer, current_seq), dim=0)

            if self.data_buffer.shape[0] <= self.bin_window_size:
                predictions.append(torch.zeros(1, self.output_dim))
            else:
                # Only pass input into model when the buffer size == bin_window_size
                if self.data_buffer.shape[0] > self.bin_window_size:
                    self.data_buffer = self.data_buffer[1:, :]

                # Accumulate
                spikes = self.data_buffer.clone()
                acc_spikes = torch.sum(spikes, dim=0)

                pred = self.single_forward(acc_spikes)
                predictions.append(pred)

        predictions = torch.stack(predictions).squeeze(dim=1)

        return predictions
    
class ANNModel3D(nn.Module):
    def __init__(self, input_dim, layer1=32, layer2=48, output_dim=2,
                 bin_window=0.2, num_steps=7, drop_rate=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = layer1
        self.layer2 = layer2
        self.drop_rate = drop_rate

        self.bin_window_time = bin_window
        self.num_steps = num_steps
        self.sampling_rate = 0.004
        self.bin_window_size = int(self.bin_window_time / self.sampling_rate)
        self.step_size = self.bin_window_size // self.num_steps

        self.fc1 = nn.Linear(self.input_dim*self.num_steps, self.layer1)
        self.fc2 = nn.Linear(self.layer1, self.layer2)
        self.fc3 = nn.Linear(self.layer2, self.output_dim)
        self.dropout = nn.Dropout(self.drop_rate)
        self.batchnorm1 = nn.BatchNorm1d(self.layer1)
        self.batchnorm2 = nn.BatchNorm1d(self.layer2)
        self.activation = nn.ReLU()

        self.register_buffer("data_buffer", torch.zeros(1, input_dim).type(torch.float32), persistent=False)

    def single_forward(self, x):
        x = x.permute(1, 0).contiguous()
        x = self.activation(self.fc1(x.view(1, -1)))
        x = self.batchnorm1(x)
        x = self.activation(self.dropout(self.fc2(x)))
        x = self.batchnorm2(x)
        x = self.fc3(x)

        return x

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
            predictions.append(pred)

        predictions = torch.stack(predictions).squeeze(dim=1)

        return predictions