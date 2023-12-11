import torch.nn as nn
import torch.nn.functional as F


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, 
                 n_channel=32, input_kernel=80, pool_kernel=4, drop=False):
        """Modified from: https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html"""
        super().__init__()

        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=input_kernel, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.2) if drop else nn.Identity()
        self.pool1 = nn.MaxPool1d(pool_kernel)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.2) if drop else nn.Identity()
        self.pool2 = nn.MaxPool1d(pool_kernel)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.2) if drop else nn.Identity()
        self.pool3 = nn.MaxPool1d(pool_kernel)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.2) if drop else nn.Identity()
        self.pool4 = nn.MaxPool1d(pool_kernel)
        self.output = nn.Linear(2 * n_channel, n_output, bias=False)


    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(self.bn1(x))
        x = self.drop1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(self.bn2(x))
        x = self.drop2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.act3(self.bn3(x))
        x = self.drop3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.act4(self.bn4(x))
        x = self.drop4(x)
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.output(x)

        return x