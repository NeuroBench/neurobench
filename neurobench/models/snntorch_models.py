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

    # TODO: Are models responsible for converting their spike trains into a
    # classification? Or do we leave the output of the models as raw spikes?
    def __call__(self, data):
        spikes = []
        utils.reset(self.net)
        for step in range(data.shape[-1]):
            spk_out, _ = self.net(data[..., step])
            spikes.append(spk_out)
        spikes = torch.stack(spikes)
        return spikes.permute(1, 2, 0)

