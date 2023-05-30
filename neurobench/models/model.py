"""
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

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

    def __init__(self, input_dim=96, layer1=32, layer2=48, output_dim=2, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm1 = nn.BatchNorm1d(layer1)
        self.batchnorm2 = nn.BatchNorm1d(layer2)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.batchnorm1(x)
        x = self.activation(self.dropout(self.fc2(x)))
        x = self.batchnorm2(x)
        x = self.fc3(x)

        return x

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