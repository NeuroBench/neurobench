import torch
import snntorch as snn
from snntorch import utils
from .neurobench_model import NeuroBenchModel


class SNNTorchModel(NeuroBenchModel):
    """The SNNTorch class wraps the forward pass of the SNNTorch framework and ensures
    that spikes are in the correct format for downstream NeuroBench components."""

    def __init__(self, net, custom_forward=False):
        """
        Init using a trained network.

        Args:
            net: A trained SNNTorch network.
            custom_forward: If True, the model's forward method is used directly. If False,
                the model is expected to take in data of shape (batch, timesteps, features*)
                and output spikes of shape (batch, timesteps, features*). Default is False.

        """
        super().__init__()

        self.net = net
        self.net.eval()

        # add snntorch neuron layers as activation modules
        self.add_activation_module(snn.SpikingNeuron)
        self.custom_forward = custom_forward

    def __call__(self, data):
        """
        Executes the forward pass of SNNTorch models on data that follows the NeuroBench
        specification. Ensures spikes are compatible with downstream components.

        Args:
            data: A PyTorch tensor of shape (batch, timesteps, ...)

        Returns:
            spikes: A PyTorch tensor of shape (batch, timesteps, ...)

        """
        if self.custom_forward:
            return self.net(data)

        # utils.reset(self.net) does not seem to delete all traces for the synaptic neuron model
        if hasattr(self.net, "reset"):
            self.net.reset()
        else:
            utils.reset(self.net)
        spikes = []

        # Data is expected to be shape (batch, timestep, features*)
        for step in range(data.shape[1]):
            out = self.net(data[:, step, ...])
            spikes.append(out[0] if isinstance(out, tuple) else out)
        spikes = torch.stack(spikes, dim=1)
        return spikes

    def __net__(self):
        """Returns the underlying network."""
        return self.net
