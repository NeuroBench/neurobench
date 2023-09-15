import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


## Define model ##
class SNN(nn.Module):

    def __init__(self, window=250, input_size=96, hidden_size=50, tau=0.96, p=0.3, device='cpu'):
        super().__init__()

        self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2
        self.surrogate = surrogate.fast_sigmoid(slope=20)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=False, device=device)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size, bias=False, device=device)

        self.lif1 = snn.Leaky(beta=tau, spike_grad=self.surrogate, threshold=1, learn_beta=False,
                              learn_threshold=False, reset_mechanism='zero')
        self.lif2 = snn.Leaky(beta=tau, spike_grad=self.surrogate, threshold=1, learn_beta=False,
                              learn_threshold=False, reset_mechanism='none')

        self.dropout = nn.Dropout(p)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        mem_rec = []

        for step in range(self.num_step):
            cur1 = self.dropout(self.fc1(x[:, :, step]))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            _, mem2 = self.lif2(cur2, mem2)

            mem_rec.append(mem2)

        if self.training:
            return torch.stack(mem_rec, dim=2)
        return mem2
