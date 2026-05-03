import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class EEG_SNN(nn.Module):
    def __init__(self, n_inputs=62, n_hidden=256, n_outputs=2, beta=0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc1  = nn.Linear(n_inputs, n_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2  = nn.Linear(n_hidden, n_hidden // 2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc3  = nn.Linear(n_hidden // 2, n_outputs)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, input):
        # input: (batch, timesteps, channels)
        _, T, _ = input.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spike_out = []

        for t in range(T):
            input_t = input[:, t, :]
            spk1, mem1 = self.lif1(self.fc1(input_t), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
            spike_out.append(spk3)

        return torch.stack(spike_out, dim=1)  # (batch, timesteps, n_outputs)