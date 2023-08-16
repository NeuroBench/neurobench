"""
"""
import torch

from snntorch import utils

from .model import NeuroBenchModel

class SNNTorchModel(NeuroBenchModel):
    """The SNNTorch class wraps the forward pass of the SNNTorch framework and 
    ensures that spikes are in the correct format for downstream NeuroBench
    components. 

    Attributes:
        net: An SNNTorch network.
    """
    def __init__(self, net):
        self.net = net
        self.net.eval()

    def __call__(self, data):
        """Executes the forward pass of SNNTorch models on data that follows the
        NeuroBench specification. Ensures spikes are compatible with downstream
        components.

        Args:
            data: A PyTorch tensor of shape (batch, timesteps, ...)

        Returns:
            spikes: A PyTorch tensor of shape (batch, timesteps, ...)
        """
        spikes = []
        utils.reset(self.net)
        spikes = []

        # Data is expected to be shape (batch, timestep, features*)
        for step in range(data.shape[1]):
            spk_out, _ = self.net(data[:, step, ...])
            spikes.append(spk_out)
        spikes = torch.stack(spikes).transpose(0, 1)
        
        return spikes
