import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer


class Conv2dLIF(nn.Module):
 
    def __init__(self, in_channels, out_channels, stride=1):
        assert out_channels % 4 == 0
        super(Conv2dLIF, self).__init__()
        
        self.conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True, step_mode='m')
        self.bn = layer.BatchNorm2d(out_channels, step_mode='m')
        self.lif = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')
        self.out_channels = out_channels

    def forward(self, x):
       
        x = self.conv1(x)
        x = self.bn(x)
        out = self.lif(x)
        return out