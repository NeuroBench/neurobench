import torch
from .model import NeuroBenchModel


class TorchModel(NeuroBenchModel):
    """The TorchModel class wraps an nn.Module."""

    def __init__(self, net):
        """
        Initializes the TorchModel class.

        Args:
            net: A PyTorch nn.Module.

        """
        super().__init__(net)

        self.net = net
        self.net.eval()

    def __call__(self, batch):
        """
        Wraps forward pass of torch.nn model.

        Args:
            batch: A PyTorch tensor of shape (batch, timesteps, features*)

        Returns:
            preds: either a tensor to be compared with targets or passed to
                NeuroBenchPostProcessors.

        """
        return self.net(batch)

    def __net__(self):
        """Returns the underlying network."""
        return self.net
