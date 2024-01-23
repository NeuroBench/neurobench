import numpy as np
import torch
import torch.nn as nn
import sys

# from metavision_core_ml.core.modules import ConvLayer, PreActBlock,ResBlock
from modules import ConvLayer, PreActBlock, ResBlock
from metavision_core_ml.core.temporal_modules import time_to_batch, SequenceWise, ConvRNN

from spikingjelly.activation_based import neuron, functional, surrogate, layer

class Conv2dLIF(nn.Module):
 
    def __init__(self, in_channels, out_channels, stride=1):
        assert out_channels % 4 == 0
        super(Conv2dLIF, self).__init__()
        
        # note: the spikingjelly.activation_based.layer.Conv2d is an instance of nn.Conv2d and is registered by the connection sparsity metric
        self.conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True, step_mode='m')
        self.bn = layer.BatchNorm2d(out_channels, step_mode='m')
        self.lif = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')
        self.out_channels = out_channels

    def forward(self, x):
       
        x = self.conv1(x)
        x = self.bn(x)
        out = self.lif(x)
        return out

class Vanilla(nn.Module):
    """
    Baseline architecture getting 0.4 mAP on the HD Event-based Automotive Detection Dataset.

    It consists of Squeeze-Excite Blocks to stride 16 and then 5 levels of Convolutional-RNNs.

    Each level can then be fed to a special head for predicting bounding boxes for example.
    """

    def __init__(self, cin=1, base=16, cout=256):
        super(Vanilla, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 5

        self.conv1 = SequenceWise(nn.Sequential(
            ConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, norm='BatchNorm2d'),
            PreActBlock(self.base * 2, self.base * 4, 2),
            PreActBlock(self.base * 4, self.base * 4, 1),
            PreActBlock(self.base * 4, self.base * 8, 1),
        ))

        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2))
        for i in range(self.levels - 1):
            self.conv2.append(ConvRNN(cout, cout, stride=2))

    def forward(self, x):
        x = self.conv1(x)
        outs = []
        for conv in self.conv2:
            x = conv(x)
            y = time_to_batch(x)[0]
            outs.append(y)
        return outs

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()

class Vanilla_lif(nn.Module):
    """
    Hybrid ANN-SNN of the above architecture, using Conv2dLIF layers instead of the ConvRNNs,
    and residual conv blocks instead of squeeze-excite blocks.
    """

    def __init__(self, cin=1, base=16, cout=256):
        super(Vanilla_lif, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 5

        self.conv1 = SequenceWise(nn.Sequential(
            ConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, norm='BatchNorm2d'),
            ResBlock(self.base * 2, self.base * 4, 2),
            ResBlock(self.base * 4, self.base * 4, 1),
            ResBlock(self.base * 4, self.base * 8, 1),
        ))

        self.conv2 = nn.ModuleList()
        self.conv2.append(Conv2dLIF(self.base * 8, cout, stride=2))
        for i in range(self.levels - 1):
            self.conv2.append(Conv2dLIF(cout, cout, stride=2))

    def forward(self, x):
        x = self.conv1(x) 
    
        outs = []
        for conv in self.conv2:
            x = conv(x)
            y = time_to_batch(x)[0]
            outs.append(y)
        return outs

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset()

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()
