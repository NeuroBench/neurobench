"""
"""
import torch
from .model import NeuroBenchModel

class TorchModel(NeuroBenchModel):
    """
    """
    def __init__(self, net):
        self.net = net
        self.net.eval()

    def __call__(self, batch):
        return self.net(batch)

    def size(self):
        param_size = 0
        for param in self.net.parameters():
            param_size += param.numel() * param.element_size()

        buffer_size = 0
        for buffer in self.net.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        return param_size + buffer_size