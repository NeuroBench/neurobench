import torch.nn as nn
import snntorch as snn
import snntorch.surrogate as surrogate

beta = 0.9
spike_grad = surrogate.fast_sigmoid()

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(20, 256),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(256, 256),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(256, 256),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(256, 35),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
)

net_rnn = nn.Sequential(
    nn.Flatten(),
    nn.RNN(input_size=20, hidden_size=20, num_layers=2, bidirectional=True),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(20, 256),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(256, 256),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(256, 35),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
)

net_relu_0 = nn.Sequential(
    nn.Identity(),
    nn.ReLU(),
)

net_relu_0_2 = nn.Sequential(
    nn.Linear(20, 25, bias=False),
    nn.Sigmoid(),
    nn.Linear(25, 25, bias=False),
    nn.ReLU(),
)

net_relu_50 = nn.Sequential(
    # nn.Flatten(),
    nn.Identity(),
    nn.ReLU(),
)

net_relu_50_2 = nn.Sequential(
    # nn.Flatten(),
    nn.Linear(20, 20, bias=False),  # 400 ops
    nn.Sigmoid(),
    nn.Linear(20, 25, bias=False),  # 500 ops
    nn.Sigmoid(),
    nn.Linear(25, 25, bias=False),  # 625 ops
    nn.Sigmoid(),
    nn.Linear(25, 25, bias=False),  # 625 ops
    nn.Sigmoid(),
)

act = nn.ReLU()
net_torch_relu_0 = nn.Sequential(
    # nn.Flatten(),
    nn.Identity(),
    act,
    nn.Identity(),
    act,
    nn.Identity(),
    act,
    nn.Identity(),
    act,
)

net_sigm = nn.Sequential(
    nn.Identity(),
    nn.Sigmoid(),
)

net_conv_2d = nn.Sequential(
    nn.Conv2d(1, 1, 3, bias=False),
    nn.ReLU(),
)

net_conv_1d = nn.Sequential(
    nn.Conv1d(5, 1, 5, bias=False),
    nn.ReLU(),
)

net_snn = nn.Sequential(
    nn.Linear(20, 5, bias=False),
    snn.Leaky(
        beta=0.9, spike_grad=surrogate.fast_sigmoid(), init_hidden=True, output=True
    ),
)


class simple_LSTM(nn.Module):
    """Nonsense LSTM for operations testing Should be 615 MACs."""

    def __init__(self):
        super(simple_LSTM, self).__init__()
        self.lstm = nn.LSTMCell(input_size=25, hidden_size=5, bias=True)
        self.rel = nn.ReLU()

    def forward(self, inp):
        x, states = inp[0], inp[1]
        x, _ = self.lstm(x, states)
        # x = self.rel(x)
        return x


class simple_RNN(nn.Module):
    """Nonsense RNN for operations testing Should be 150 MACs."""

    def __init__(self):
        super(simple_RNN, self).__init__()
        self.RNN = nn.RNNCell(input_size=25, hidden_size=5, bias=True)
        self.rel = nn.ReLU()

    def forward(self, inp):
        x, states = inp[0], inp[1]
        x = self.RNN(x, states)
        x = self.rel(x)
        return x


class simple_GRU(nn.Module):
    """Nonsense GRU/RNN for operations testing Should be 465 MACs."""

    def __init__(self):
        super(simple_GRU, self).__init__()
        self.GRU = nn.GRUCell(input_size=25, hidden_size=5, bias=True)
        self.rel = nn.ReLU()

    def forward(self, inp):
        x, states = inp[0], inp[1]
        x = self.GRU(x, states)
        x = self.rel(x)
        return x
