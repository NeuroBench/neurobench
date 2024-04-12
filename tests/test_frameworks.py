import pytest

import torch
import snntorch as snn

from torch import nn
from snntorch import surrogate

from neurobench.models import SNNTorchModel


def test_snntorch_framework():
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
    model = SNNTorchModel(net)

    # Random data of shape (batch=256, ts=1000, chan=20)
    data = torch.rand((256, 1000, 20))
    spikes = model(data)
    assert data.shape == (256, 1000, 20)
    assert spikes.shape == (256, 1000, 35)

    data = torch.rand((256, 1000, 2, 2, 5))
    spikes = model(data)
    assert data.shape == (256, 1000, 2, 2, 5)
    assert spikes.shape == (256, 1000, 35)

    data = torch.rand((256, 1000, 10, 5))
    with pytest.raises(RuntimeError, match="mat1 and mat2 shapes cannot be multiplied"):
        spikes = model(data)
