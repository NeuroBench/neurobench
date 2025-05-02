import torch
import torch.nn as nn

class ANNModel(nn.Module):
    def __init__(
        self, 
        input_dim, 
        layer1=16, 
        layer2=32, 
        output_dim=2,
        bin_window=0.2, 
        sampling_rate=0.004, 
        drop_rate=0.5
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = layer1
        self.layer2 = layer2
        self.drop_rate = drop_rate

        self.bin_window_time = bin_window
        self.sampling_rate = sampling_rate
        self.bin_window_size = int(self.bin_window_time / self.sampling_rate)

        self.fc1 = nn.Linear(self.input_dim, self.layer1)
        self.fc2 = nn.Linear(self.layer1, self.layer2)
        self.fc3 = nn.Linear(self.layer2, self.output_dim)
        self.dropout = nn.Dropout(self.drop_rate)
        self.batchnorm1 = nn.BatchNorm1d(self.layer1)
        self.batchnorm2 = nn.BatchNorm1d(self.layer2)
        self.activation = nn.ReLU()
        self.batch_size = 1

        self.register_buffer("data_buffer", torch.zeros(1, 1, input_dim).type(torch.float32), persistent=False)

    def forward(self, x):
        self.data_buffer = torch.cat((self.data_buffer, x), dim=0)
        self.data_buffer = self.data_buffer[1:, :, :]

        x = self.activation(self.fc1(x.view(self.batch_size, -1)))
        x = self.activation(self.dropout(self.fc2(x)))
        x = self.fc3(x)

        pred = x.squeeze(dim=0)

        return pred