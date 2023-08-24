import torch
from .model import NeuroBenchModel

class TorchModel(NeuroBenchModel):
    """ The TorchModel class wraps an nn.Module.
    """
    def __init__(self, net):
        """ Initializes the TorchModel class.

        Args:
            net: A PyTorch nn.Module.
        """
        self.net = net
        self.net.eval()

    def __call__(self, batch):
        """ Wraps forward pass of torch.nn model.

        Args:
            batch: A PyTorch tensor of shape (batch, timesteps, features*)

        Returns:
            preds: either a tensor to be compared with targets or passed to
                NeuroBenchAccumulators.
        """
        return self.net(batch)

    def size(self):
        """ Model footprint in bytes.

        Returns:
            size: An int representing the size of the model in bytes.
        """
        param_size = 0
        for param in self.net.parameters():
            param_size += param.numel() * param.element_size()

        buffer_size = 0
        for buffer in self.net.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        return param_size + buffer_size