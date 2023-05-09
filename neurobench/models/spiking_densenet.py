from collections import OrderedDict
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, neuron
from torchvision._internally_replaced_utils import load_state_dict_from_url

__all__ = [
    "SpikingDenseNet", "MultiStepSpikingDenseNet", 
    "spiking_densenet121", "multi_step_spiking_densenet121",
    "spiking_densenet169", "multi_step_spiking_densenet169", 
    "spiking_densenet201", "multi_step_spiking_densenet201", 
    "spiking_densenet161", "multi_step_spiking_densenet161",
    "multi_step_spiking_densenet_custom",
]

# modified from https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, 
                 norm_layer: callable, bias: bool, neuron: callable, **kwargs):
        super().__init__()
        self.drop_rate = float(drop_rate)
        
        self.norm1 = norm_layer(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=bias)
        self.act1 = neuron(**kwargs)

        self.norm2 = norm_layer(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=bias)
        self.act2 = neuron(**kwargs)

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.act1(self.conv1(self.norm1(concated_features))) 
        return bottleneck_output

    def forward(self, input): 
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)
        
        new_features = self.act2(self.conv2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = nn.functional.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers: int, num_input_features: int, bn_size: int, 
                 growth_rate: int, drop_rate: float, norm_layer: callable, bias: bool, 
                 neuron: callable = None, **kwargs):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                norm_layer=norm_layer,
                neuron=neuron,
                bias=bias,
                **kwargs
            )
            self.add_module(f"denselayer{i + 1}", layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, 
                 norm_layer: callable, bias: bool, neuron: callable = None, **kwargs):
        super().__init__()
        self.add_module("norm", norm_layer(num_input_features))
        self.add_module("conv", nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=bias))
        self.add_module("act", neuron(**kwargs))
        self.add_module("pool", nn.MaxPool2d(kernel_size=2, stride=2))

class SpikingDenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), 
                 num_init_channels=2, bn_size=4, drop_rate=0, 
                 num_classes=1000, init_weights=True, norm_layer: callable = None, 
                 neuron: callable = None, **kwargs):
        
        super().__init__()
        
        self.nz, self.numel = {}, {}
        self.out_channels = []
        
        if norm_layer is None:
            norm_layer = nn.Identity
        bias = isinstance(norm_layer, nn.Identity)
            
        num_init_features = 2 * growth_rate

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("pad0", nn.ConstantPad2d(1, 0.)),
                    ("norm0", norm_layer(num_init_channels)),
                    ("conv0", nn.Conv2d(num_init_channels, num_init_features, 
                                        kernel_size=3, stride=2, padding=0, bias=bias)),
                    ("act0", neuron(**kwargs)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                norm_layer=norm_layer,
                bias=bias,
                neuron=neuron,
                **kwargs
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate
            
            # register feature maps size after trans1, trans2, dense4 (not after trans3) for OD
            if i != len(block_config) - 2:
                self.out_channels.append(num_features)
                
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features, 
                    num_output_features=num_features // 2, 
                    norm_layer=norm_layer,
                    bias=bias,
                    neuron=neuron,
                    **kwargs
                )
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2
                
        
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("norm_classif", norm_layer(num_features)),
                    ("conv_classif", nn.Conv2d(num_features, num_classes, 
                                                kernel_size=1, bias=bias)),
                    ("act_classif", neuron(**kwargs)),
                ]
            )
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        self.reset_nz_numel()
        features = self.features(x)
        out = self.classifier(features)
        out = out.flatten(start_dim=-2).sum(dim=-1)
        return out
    
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


class MultiStepSpikingDenseNet(SpikingDenseNet):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), 
                 num_init_channels=2, bn_size=4, drop_rate=0, 
                 num_classes=1000, init_weights=True, norm_layer: callable = None,
                 T=None, neuron: callable = None, **kwargs):
        self.T = T
        super().__init__(growth_rate, block_config, num_init_channels, bn_size, drop_rate, num_classes, init_weights, norm_layer, neuron, **kwargs)
    
    def forward(self, x, classify=True):
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
            
        if classify:
            x_seq = sequential_forward(self.features[1:], x_seq)
            x_seq = functional.seq_to_ann_forward(x_seq, self.classifier)
            x_seq = x_seq.flatten(start_dim=-2).sum(dim=-1)
            return x_seq
        else:
            fm_trans1 = sequential_forward(self.features[1:6], x_seq)
            fm_trans2 = sequential_forward(self.features[6:8], fm_trans1)
            x_seq = sequential_forward(self.features[8:], fm_trans2)
            return fm_trans1, fm_trans2, x_seq

def _densenet(
    arch: str,
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_channels: int,
    norm_layer: callable = None, single_step_neuron: callable = None,
    **kwargs: Any,
) -> SpikingDenseNet:
    model = SpikingDenseNet(growth_rate, block_config, num_init_channels, norm_layer=norm_layer, neuron=single_step_neuron, **kwargs)
    return model


def _multi_step_densenet(
    arch: str,
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_channels: int,
    norm_layer: callable = None, T=None, multi_step_neuron: callable = None,
    **kwargs: Any,
) -> SpikingDenseNet:
    model = MultiStepSpikingDenseNet(growth_rate, block_config, num_init_channels, norm_layer=norm_layer, T=T, neuron=multi_step_neuron, **kwargs)
    return model

def spiking_densenet_custom(num_init_channels, norm_layer: callable = None, single_step_neuron: callable = None, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs) -> SpikingDenseNet:
    r"""A spiking version of custom DenseNet model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        num_init_channels (int): number of channels of the input data
        norm_layer (callable): a layer of batch norm. No batch norm if None
    """
    return _densenet("densenet", growth_rate, block_config, num_init_channels, norm_layer, single_step_neuron, **kwargs)


def multi_step_spiking_densenet_custom(num_init_channels, norm_layer: callable = None, T=None, multi_step_neuron: callable = None, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs) -> SpikingDenseNet:
    r"""A multi-step spiking version of custom DenseNet model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        num_init_channels (int): number of channels of the input data
        norm_layer (callable): a layer of batch norm. No batch norm if None
    """
    return _multi_step_densenet("densenet", growth_rate, block_config, num_init_channels, norm_layer, T, multi_step_neuron, **kwargs)

def spiking_densenet121(num_init_channels, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs) -> SpikingDenseNet:
    r"""A spiking version of Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        num_init_channels (int): number of channels of the input data
        norm_layer (callable): a layer of batch norm. No batch norm if None
    """
    return _densenet("densenet121", 32, (6, 12, 24, 16), num_init_channels, norm_layer, single_step_neuron, **kwargs)


def multi_step_spiking_densenet121(num_init_channels, norm_layer: callable = None, T=None, multi_step_neuron: callable = None, **kwargs) -> SpikingDenseNet:
    r"""A multi-step spiking version of Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        num_init_channels (int): number of channels of the input data
        norm_layer (callable): a layer of batch norm. No batch norm if None
    """
    return _multi_step_densenet("densenet121", 32, (6, 12, 24, 16), num_init_channels, norm_layer, T, multi_step_neuron, **kwargs)


def spiking_densenet161(num_init_channels, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs) -> SpikingDenseNet:
    r"""A spiking version of Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        num_init_channels (int): number of channels of the input data
        norm_layer (callable): a layer of batch norm. No batch norm if None
    """
    return _densenet("densenet161", 48, (6, 12, 36, 24), num_init_channels, norm_layer, single_step_neuron, single_step_neuron, **kwargs)


def multi_step_spiking_densenet161(num_init_channels, norm_layer: callable = None, T=None, multi_step_neuron: callable = None, **kwargs) -> SpikingDenseNet:
    r"""A multi-step spiking version of Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        num_init_channels (int): number of channels of the input data
        norm_layer (callable): a layer of batch norm. No batch norm if None
    """
    return _multi_step_densenet("densenet161", 48, (6, 12, 36, 24), num_init_channels, norm_layer, T, multi_step_neuron, **kwargs)


def spiking_densenet169(num_init_channels, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs) -> SpikingDenseNet:
    r"""A spiking version of Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        num_init_channels (int): number of channels of the input data
        norm_layer (callable): a layer of batch norm. No batch norm if None
    """
    return _densenet("densenet169", 32, (6, 12, 32, 32), num_init_channels, norm_layer, single_step_neuron, **kwargs)


def multi_step_spiking_densenet169(num_init_channels, norm_layer: callable = None, T=None, multi_step_neuron: callable = None, **kwargs) -> SpikingDenseNet:
    r"""A multi-step spiking version of Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        num_init_channels (int): number of channels of the input data
        norm_layer (callable): a layer of batch norm. No batch norm if None
    """
    return _multi_step_densenet("densenet169", 32, (6, 12, 32, 32), num_init_channels, norm_layer, T, multi_step_neuron, **kwargs)


def spiking_densenet201(num_init_channels, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs) -> SpikingDenseNet:
    r"""A spiking version of Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        num_init_channels (int): number of channels of the input data
        norm_layer (callable): a layer of batch norm. No batch norm if None
    """
    return _densenet("densenet201", 32, (6, 12, 48, 32), num_init_channels, norm_layer, single_step_neuron, **kwargs)


def multi_step_spiking_densenet201(num_init_channels, norm_layer: callable = None, T=None, multi_step_neuron: callable = None, **kwargs) -> SpikingDenseNet:
    r"""A multi-step spiking version of Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        num_init_channels (int): number of channels of the input data
        norm_layer (callable): a layer of batch norm. No batch norm if None
    """
    return _multi_step_densenet("densenet201", 32, (6, 12, 48, 32), num_init_channels, norm_layer, T, multi_step_neuron, **kwargs)