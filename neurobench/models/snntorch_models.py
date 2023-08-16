"""
"""
import torch

from snntorch import utils

from .model import NeuroBenchModel

class SNNTorchModel(NeuroBenchModel):
    """
    """
    def __init__(self, net):
        self.net = net
        self.net.eval()

    def __call__(self, data):
        utils.reset(self.net)
        spikes = []
        # Data is expected to be shape (batch, timestep, features*)
        for step in range(data.shape[1]):
            spk_out, _ = self.net(data[:, step, ...])
            spikes.append(spk_out)
        spikes = torch.stack(spikes).transpose(0, 1)
        return spikes
