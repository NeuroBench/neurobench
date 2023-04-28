import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class SnnModel(nn.Module):
    """
    A 3-layer SNN with leaky-integrate-fire neuron without reset_mechanism in last layer

    :param input_dim: input feature dimension
    :param layer1: Number of hidden neurons in the first layer
    :param layer2: Number of hidden neurons in the second layer
    :param output_dim: output feature dimension
    :param beta: Membrane potential decay rate
    :param spike_grad: Surrogate gradient
    :param mem_threshold: membrane threshold
    :param num_step: Number of time steps
    :param dropout_rate: Probability of dropout
    """
    def __init__(
            self,
            input_dim=128,
            layer1=128,
            layer2=64,
            output_dim=2,
            beta=0.9,
            spike_grad=surrogate.fast_sigmoid(slope=20),
            mem_threshold=1,
            num_step=5,
            dropout_rate=0.5):

        super().__init__()

        self.num_step = num_step

        self.fc1 = nn.Linear(input_dim, layer1)
        self.fc3 = nn.Linear(layer1, layer2)
        self.fc4 = nn.Linear(layer2, output_dim)

        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                              threshold=mem_threshold)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                              threshold=mem_threshold)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=mem_threshold, learn_beta=True,
                              reset_mechanism="none")

        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm1 = nn.BatchNorm1d(layer1)
        self.batchnorm2 = nn.BatchNorm1d(layer2)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk_rec = []
        mem_rec = []

        for step in range(self.num_step):
            cur1 = self.batchnorm1(self.dropout(self.fc1(x)))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur3 = self.batchnorm2(self.dropout(self.fc3(spk1)))
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk_rec.append(spk4)
            mem_rec.append(mem4)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
