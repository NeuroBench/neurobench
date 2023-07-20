
import torch
import torch.nn as nn

from spikingjelly.activation_based.layer import Conv2d, BatchNorm2d
from spikingjelly.activation_based import surrogate, neuron
from metavision_core_ml.core.temporal_modules import time_to_batch


def _xavier_init(block):
    if isinstance(block, nn.Sequential):
        for layer in block.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, torch.tensor(0, dtype=torch.float16))
    if isinstance(block, list):
        for inner_block in block:
            _xavier_init(inner_block)

class BackBone_OD(nn.Module):
    def __init__(self, cin=6, base=16, cout=64):
        super(BackBone_OD, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 6
        self.out_channels_list = []

        self.conv_block1 = nn.Sequential(
            Conv2d(cin, self.base * 2, kernel_size=7, stride=2, padding=3, step_mode='m'),
            BatchNorm2d(self.base * 2, step_mode='m'),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')
        )

        conv = []
        for level in range(self.levels):
            if conv.__len__() == 0:
                in_channels = self.base * 2
                cout = cout
            else:
                in_channels = cout
                cout = self.base * 4 * level
            self.out_channels_list.append(cout)
            conv.append(
                nn.Sequential(
                    Conv2d(in_channels, cout, kernel_size=3, padding=1, bias=True, step_mode='m'),
                    BatchNorm2d(cout, step_mode='m'),
                    neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m')
                )
            )


        self.conv_block2 = conv
        _xavier_init(self.conv_block1)
        _xavier_init(self.conv_block2)


    def forward(self, x):
        x = self.conv_block1(x) 
    
        outs = []
        for block in self.conv_block2:
            x = block.to('cuda')(x)
            y = time_to_batch(x)[0]
            outs.append(y)
        return outs

    def reset(self, mask=None):
        for module in self.conv_block1.modules():
            if hasattr(module, "reset"):
                module.reset()
        for block in self.conv_block2:
            for module in block:
                if hasattr(module, "reset"):
                    module.reset()

    @torch.jit.export
    def reset_all(self):
        for module in self.conv_block1.modules:
            module.reset_all()
        for block in self.conv_block2:
            for module in block:
                module.reset_all()