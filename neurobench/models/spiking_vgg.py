from collections import OrderedDict

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, neuron

__all__ = [
    'SpikingVGG', 'MultiStepSpikingVGG',
    'multi_step_spiking_vgg11','spiking_vgg11',
    'multi_step_spiking_vgg13','spiking_vgg13',
    'multi_step_spiking_vgg16','spiking_vgg16',
    'multi_step_spiking_vgg19','spiking_vgg19',
    'multi_step_spiking_vgg_custom','spiking_vgg_custom',
]

# modified from https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py

class SpikingVGG(nn.Module):
    def __init__(self, num_init_channels, cfg, norm_layer=None, num_classes=1000, init_weights=True,
                 single_step_neuron: callable = None, **kwargs):
        super(SpikingVGG, self).__init__()
        
        self.nz, self.numel = {}, {}
        self.out_channels = []
        self.idx_pool = [i for i,v in enumerate(cfg) if v=='M']

        if norm_layer is None:
            norm_layer = nn.Identity
        bias = isinstance(norm_layer, nn.Identity)
        
        self.features = self.make_layers(num_init_channels, cfg=cfg,
                                         norm_layer=norm_layer, neuron=single_step_neuron, 
                                         bias=bias,**kwargs)
        
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("norm_classif", norm_layer(cfg[-2])),
                    ("conv_classif", nn.Conv2d(cfg[-2], num_classes, 
                                                kernel_size=1, bias=bias)),
                    ("act_classif", single_step_neuron(**kwargs)),
                ]
            )
        )
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        self.reset_nz_numel()
        x = self.features(x)
        x = self.classifier(x)
        x = x.flatten(start_dim=-2).sum(dim=-1)
        return x

    def make_layers(self, num_init_channels, cfg, norm_layer, neuron, bias, **kwargs):
        layers = []
        in_channels = num_init_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                self.out_channels.append(in_channels)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=0, bias=bias)
                layers += [nn.Sequential(
                    OrderedDict(
                        [
                            ("padding", nn.ConstantPad2d(1, 0.)),
                            ("norm", norm_layer(in_channels)),
                            ("conv", conv2d),
                            ("act", neuron(**kwargs)),
                        ]
                    )
                )]
                in_channels = v
        self.out_channels = self.out_channels[2:]
        return nn.Sequential(*layers)

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


class MultiStepSpikingVGG(SpikingVGG):
    def __init__(self, num_init_channels, cfg, norm_layer=None, num_classes=1000, init_weights=True, T: int = None,
                 multi_step_neuron: callable = None, **kwargs):
        self.T = T
        super().__init__(num_init_channels, cfg, norm_layer, num_classes, init_weights,
                 multi_step_neuron, **kwargs)

    def forward(self, x, classify=True):
        x_seq = None
        if x.dim() == 5:
            # x.shape = [T, N, C, H, W]
            x_seq = functional.seq_to_ann_forward(x, self.features[0])
        else:
            assert self.T is not None, 'When x.shape is [N, C, H, W], self.T can not be None.'
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
            fm_1 = sequential_forward(self.features[1:self.idx_pool[2]], x_seq)
            fm_2 = sequential_forward(self.features[self.idx_pool[2]:self.idx_pool[3]], fm_1)
            x_seq = sequential_forward(self.features[self.idx_pool[3]:], fm_2)
            return fm_1, fm_2, x_seq


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _spiking_vgg(arch, cfg, num_init_channels, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs):
    return SpikingVGG(num_init_channels, cfg=cfg, norm_layer=norm_layer, single_step_neuron=single_step_neuron, **kwargs)

def _multi_step_spiking_vgg(arch, cfg, num_init_channels, norm_layer: callable = None, T: int = None, multi_step_neuron: callable = None, **kwargs):
    return MultiStepSpikingVGG(num_init_channels, cfg=cfg, norm_layer=norm_layer, T=T, multi_step_neuron=multi_step_neuron, **kwargs)

def spiking_vgg_custom(num_init_channels: int, cfg, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs):
    """
    :param num_init_channels: number of channels of the input data
    :type num_init_channels: int
    :param cfg: configuration of the VGG layers (num channels, pooling)
    :type cfg: list
    :param norm_layer: a batch norm layer
    :type norm_layer: callable
    :param single_step_neuron: a single-step neuron
    :type single_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking VGG-11 with norm layer
    :rtype: torch.nn.Module
    A spiking version of VGG-11-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg_custom', cfg, num_init_channels, norm_layer, single_step_neuron, **kwargs)


def multi_step_spiking_vgg_custom(num_init_channels, cfg, norm_layer: callable = None, T: int = None, multi_step_neuron: callable = None, **kwargs):
    """
    :param num_init_channels: number of channels of the input data
    :type num_init_channels: int
    :param cfg: configuration of the VGG layers (num channels, pooling)
    :type cfg: list
    :param norm_layer: a batch norm layer
    :type norm_layer: callable
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi_step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking VGG-11 with norm layer
    :rtype: torch.nn.Module
    A multi-step spiking version of VGG-11-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _multi_step_spiking_vgg('vgg_custom', cfg, num_init_channels, norm_layer, T, multi_step_neuron, **kwargs)

def spiking_vgg11(num_init_channels: int, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs):
    """
    :param num_init_channels: number of channels of the input data
    :type num_init_channels: int
    :param norm_layer: a batch norm layer
    :type norm_layer: callable
    :param single_step_neuron: a single-step neuron
    :type single_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking VGG-11 with norm layer
    :rtype: torch.nn.Module
    A spiking version of VGG-11-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg11', cfgs['A'], num_init_channels, norm_layer, single_step_neuron, **kwargs)


def multi_step_spiking_vgg11(num_init_channels, norm_layer: callable = None, T: int = None, multi_step_neuron: callable = None, **kwargs):
    """
    :param num_init_channels: number of channels of the input data
    :type num_init_channels: int
    :param norm_layer: a batch norm layer
    :type norm_layer: callable
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi_step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking VGG-11 with norm layer
    :rtype: torch.nn.Module
    A multi-step spiking version of VGG-11-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _multi_step_spiking_vgg('vgg11', cfgs['A'], num_init_channels, norm_layer, T, multi_step_neuron, **kwargs)


def spiking_vgg13(num_init_channels: int, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs):
    """
    :param num_init_channels: number of channels of the input data
    :type num_init_channels: int
    :param norm_layer: a batch norm layer
    :type norm_layer: callable
    :param single_step_neuron: a single-step neuron
    :type single_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking VGG-13 with norm layer
    :rtype: torch.nn.Module
    A spiking version of VGG-13-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg13', cfgs['B'], num_init_channels, norm_layer, single_step_neuron, **kwargs)


def multi_step_spiking_vgg13(num_init_channels: int, norm_layer: callable = None, T: int = None, multi_step_neuron: callable = None, **kwargs):
    """
    :param num_init_channels: number of channels of the input data
    :type num_init_channels: int
    :param norm_layer: a batch norm layer
    :type norm_layer: callable
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi_step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking VGG-13 with norm layer
    :rtype: torch.nn.Module
    A multi-step spiking version of VGG-13-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _multi_step_spiking_vgg('vgg13', cfgs['B'], num_init_channels, norm_layer, T, multi_step_neuron, **kwargs)


def spiking_vgg16(num_init_channels: int, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs):
    """
    :param num_init_channels: number of channels of the input data
    :type num_init_channels: int
    :param norm_layer: a batch norm layer
    :type norm_layer: callable
    :param single_step_neuron: a single-step neuron
    :type single_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking VGG-16 with norm layer
    :rtype: torch.nn.Module
    A spiking version of VGG-16-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg16', cfgs['D'], num_init_channels, norm_layer, single_step_neuron, **kwargs)


def multi_step_spiking_vgg16(num_init_channels: int, norm_layer: callable = None, T: int = None, multi_step_neuron: callable = None, **kwargs):
    """
    :param num_init_channels: number of channels of the input data
    :type num_init_channels: int
    :param norm_layer: a batch norm layer
    :type norm_layer: callable
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi_step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking VGG-16 with norm layer
    :rtype: torch.nn.Module
    A multi-step spiking version of VGG-16-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _multi_step_spiking_vgg('vgg16', cfgs['D'], num_init_channels, norm_layer, T, multi_step_neuron, **kwargs)


def spiking_vgg19(num_init_channels: int, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs):
    """
    :param num_init_channels: number of channels of the input data
    :type num_init_channels: int
    :param norm_layer: a batch norm layer
    :type norm_layer: callable
    :param single_step_neuron: a single-step neuron
    :type single_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking VGG-19 with norm layer
    :rtype: torch.nn.Module
    A spiking version of VGG-19-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _spiking_vgg('vgg19', cfgs['E'], num_init_channels, norm_layer, single_step_neuron, **kwargs)


def multi_step_spiking_vgg19(num_init_channels: int, norm_layer: callable = None, T: int = None, multi_step_neuron: callable = None, **kwargs):
    """
    :param num_init_channels: number of channels of the input data
    :type num_init_channels: int
    :param norm_layer: a batch norm layer
    :type norm_layer: callable
    :param T: total time-steps
    :type T: int
    :param multi_step_neuron: a multi_step neuron
    :type multi_step_neuron: callable
    :param kwargs: kwargs for `single_step_neuron`
    :type kwargs: dict
    :return: Spiking VGG-19 with norm layer
    :rtype: torch.nn.Module
    A multi-step spiking version of VGG-19-BN model from `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """

    return _multi_step_spiking_vgg('vgg19', cfgs['E'], num_init_channels, norm_layer, T, multi_step_neuron, **kwargs)