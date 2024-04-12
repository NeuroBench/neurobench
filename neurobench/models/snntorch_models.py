import torch

import snntorch as snn
from snntorch import utils

from .model import NeuroBenchModel


class SNNTorchModel(NeuroBenchModel):
    """The SNNTorch class wraps the forward pass of the SNNTorch framework and ensures
    that spikes are in the correct format for downstream NeuroBench components."""

    def __init__(self, net):
        """
        Init using a trained network.

        Args:
            net: A trained SNNTorch network.

        """
        super().__init__(net)

        self.net = net
        self.net.eval()

        # add snntorch neuron layers as activation modules
        self.add_activation_module(snn.SpikingNeuron)

    def __call__(self, data):
        """
        Executes the forward pass of SNNTorch models on data that follows the NeuroBench
        specification. Ensures spikes are compatible with downstream components.

        Args:
            data: A PyTorch tensor of shape (batch, timesteps, ...)

        Returns:
            spikes: A PyTorch tensor of shape (batch, timesteps, ...)

        """
        spikes = []
        # utils.reset(self.net) does not seem to delete all traces for the synaptic neuron model
        if hasattr(self.net, "reset"):
            self.net.reset()
        else:
            utils.reset(self.net)
        spikes = []

        # Data is expected to be shape (batch, timestep, features*)
        for step in range(data.shape[1]):
            spk_out, _ = self.net(data[:, step, ...])
            spikes.append(spk_out)
        spikes = torch.stack(spikes).transpose(0, 1)
        return spikes

    def __net__(self):
        """Returns the underlying network."""
        return self.net
