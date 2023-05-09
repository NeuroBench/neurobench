from typing import Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init
from spikingjelly.clock_driven import functional, neuron

__all__ = ["SpikingSqueezeNet", "MultiStepSpikingSqueezeNet",
           "spiking_squeezenet1_0", "multi_step_spiking_squeezenet1_0",
           "spiking_squeezenet1_1", "multi_step_spiking_squeezenet1_1"]

# modified from https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py

class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int, norm_layer: callable, bias: bool, neuron: callable = None, **kwargs):
        super().__init__()
        self.inplanes = inplanes
        
        self.squeeze_norm = norm_layer(inplanes)
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, bias=bias)
        self.squeeze_activation = neuron(**kwargs)
        
        self.expand_norm = norm_layer(squeeze_planes)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1, bias=bias)
        self.expand1x1_activation = neuron(**kwargs)
        
        self.pad_before_expand3x3 = nn.ConstantPad2d(1, 0.)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=0, bias=bias)
        self.expand3x3_activation = neuron(**kwargs)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(self.squeeze_norm(x)))
        
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(self.expand_norm(x))), 
            self.expand3x3_activation(self.expand3x3(self.expand_norm(self.pad_before_expand3x3(x))))
        ], 1)


class SpikingSqueezeNet(nn.Module):
    def __init__(self, version: str = "1_0", num_init_channels=2, num_classes: int = 1000, dropout: float = 0.5, init_weights=True, norm_layer: callable = None, neuron: callable = None, **kwargs):
        super().__init__()
        
        self.nz, self.numel = {}, {}
        
        if norm_layer is None:
            norm_layer = nn.Identity
        bias = isinstance(norm_layer, nn.Identity)
            
        if version == "1_0":
            self.features = nn.Sequential(
                nn.Sequential(OrderedDict(
                    [
                        ("pad0", nn.ConstantPad2d(3, 0.)),
                        ("norm0", norm_layer(num_init_channels)),
                        ("conv0", nn.Conv2d(num_init_channels, 96, 
                                            kernel_size=7, stride=2, padding=0, bias=bias)),
                        ("act0", neuron(**kwargs)),
                        ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    ]
                )),
                Fire(96, 16, 64, 64, norm_layer, bias, neuron, **kwargs),
                Fire(128, 16, 64, 64, norm_layer, bias, neuron, **kwargs),
                Fire(128, 32, 128, 128, norm_layer, bias, neuron, **kwargs),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                Fire(256, 32, 128, 128, norm_layer, bias, neuron, **kwargs),
                Fire(256, 48, 192, 192, norm_layer, bias, neuron, **kwargs),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # added
                Fire(384, 48, 192, 192, norm_layer, bias, neuron, **kwargs),
                Fire(384, 64, 256, 256, norm_layer, bias, neuron, **kwargs),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                Fire(512, 64, 256, 256, norm_layer, bias, neuron, **kwargs),
            )
        elif version == "1_1":
            self.features = nn.Sequential(
                nn.Sequential(OrderedDict(
                    [
                        ("pad0", nn.ConstantPad2d(1, 0.)),
                        ("norm0", norm_layer(num_init_channels)),
                        ("conv0", nn.Conv2d(num_init_channels, 64, 
                                            kernel_size=3, stride=2, padding=0, bias=bias)),
                        ("act0", neuron(**kwargs)),
                        ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    ]
                )),
                Fire(64, 16, 64, 64, norm_layer, bias, neuron, **kwargs),
                Fire(128, 16, 64, 64, norm_layer, bias, neuron, **kwargs),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                Fire(128, 32, 128, 128, norm_layer, bias, neuron, **kwargs),
                Fire(256, 32, 128, 128, norm_layer, bias, neuron, **kwargs),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                Fire(256, 48, 192, 192, norm_layer, bias, neuron, **kwargs),
                Fire(384, 48, 192, 192, norm_layer, bias, neuron, **kwargs),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # added
                Fire(384, 64, 256, 256, norm_layer, bias, neuron, **kwargs),
                Fire(512, 64, 256, 256, norm_layer, bias, neuron, **kwargs),
            )

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("norm_classif", norm_layer(512)),
                    ("conv_classif", nn.Conv2d(512, num_classes, 
                                                kernel_size=1, bias=bias)),
                    ("act_classif", neuron(**kwargs)),
                ]
            )
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.reset_nz_numel()
        x = self.features(x)
        x = self.classifier(x)
        x = x.flatten(start_dim=-2).sum(dim=-1)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def add_hooks(self):
        def get_nz(name):
            def hook(model, input, output):
                self.nz[name] += torch.count_nonzero(output)
                self.numel[name] += output.numel()
            return hook
        
        self.hooks = {}
        for name, module in self.named_modules():
            self.nz[name], self.numel[name] = 0, 0
            self.hooks[name] = module.register_forward_hook(get_nz(name))
                
    def reset_nz_numel(self):
        for name, module in self.named_modules():
            self.nz[name], self.numel[name] = 0, 0
        
    def get_nz_numel(self):
        return self.nz, self.numel
    
    
def sequential_forward(sequential, x_seq):
    assert isinstance(sequential, nn.Sequential)
    out = x_seq
    for i in range(len(sequential)):
        m = sequential[i]
        if isinstance(m, neuron.BaseNode):
            out = m(out)
        else:
            out = functional.seq_to_ann_forward(out, m)
    return out    


class MultiStepSpikingSqueezeNet(SpikingSqueezeNet):
    def __init__(self, version: str = "1_0", num_init_channels=2, num_classes: int = 1000, dropout: float = 0.5, init_weights=True, norm_layer: callable = None, T=None, neuron: callable = None, **kwargs):
        self.T = T
        super().__init__(version, num_init_channels, num_classes, dropout, init_weights, norm_layer, neuron, **kwargs)

    def forward(self, x):
        x_seq = None
        if x.dim() == 5:
            # x.shape = [T, N, C, H, W]
            x_seq = functional.seq_to_ann_forward(x, self.features[0])
        else:
            assert self.T is not None, "When x.shape is [N, C, H, W], self.T can not be None."
            # x.shape = [N, C, H, W]
            x = self.features[0](x)
            x.unsqueeze_(0)
            x_seq = x.repeat(self.T, 1, 1, 1, 1)
        x_seq = sequential_forward(self.features[1:], x_seq)
        x_seq = functional.seq_to_ann_forward(x_seq, self.classifier)
        x_seq = x_seq.flatten(start_dim=-2).sum(dim=-1)
        return x_seq


def _spiking_squeezenet(version: str, 
    num_init_channels: int, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs: Any):
    return SpikingSqueezeNet(version, num_init_channels, norm_layer=norm_layer, neuron=single_step_neuron, **kwargs)

def _multi_step_spiking_squeezenet(version: str, 
    num_init_channels: int, norm_layer: callable = None, T=None, multi_step_neuron: callable = None, **kwargs):
    return MultiStepSpikingSqueezeNet(version, num_init_channels, norm_layer=norm_layer, T=T, neuron=multi_step_neuron,**kwargs)


def spiking_squeezenet1_0(num_init_channels, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs: Any):
    r"""A spiking version of SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    The required minimum input size of the model is 21x21.
    Args:
        num_init_channels (int): number of channels of the input data
        norm_layer (callable): a layer of batch norm. No batch norm if None
    """
    return _spiking_squeezenet("1_0", num_init_channels, norm_layer, single_step_neuron, **kwargs)

def multi_step_spiking_squeezenet1_0(num_init_channels, norm_layer: callable = None, T=None, multi_step_neuron: callable = None, **kwargs: Any):
    r"""A multi-step spiking version of SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    The required minimum input size of the model is 21x21.
    Args:
        num_init_channels (int): number of channels of the input data
        norm_layer (callable): a layer of batch norm. No batch norm if None
    """
    return _multi_step_spiking_squeezenet("1_0", num_init_channels, norm_layer, T, multi_step_neuron, **kwargs)


def spiking_squeezenet1_1(num_init_channels, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs: Any):
    r"""A spiking version of SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    The required minimum input size of the model is 17x17.
    Args:
        num_init_channels (int): number of channels of the input data
        norm_layer (callable): a layer of batch norm. No batch norm if None
    """
    return _spiking_squeezenet("1_1", num_init_channels, norm_layer, single_step_neuron, **kwargs)

def multi_step_spiking_squeezenet1_1(num_init_channels, norm_layer: callable = None, T=None, multi_step_neuron: callable = None, **kwargs: Any):
    r"""A multi-step spiking version of SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    The required minimum input size of the model is 17x17.
    Args:
        num_init_channels (int): number of channels of the input data
        norm_layer (callable): a layer of batch norm. No batch norm if None
    """
    return _multi_step_spiking_squeezenet("1_1", num_init_channels, norm_layer, T, multi_step_neuron, **kwargs)