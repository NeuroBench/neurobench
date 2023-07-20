from collections import OrderedDict

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, neuron
from metavision_core_ml.core.temporal_modules import time_to_batch

from .vgg_utils import SpikingBlock, SpikingVGG, sequential_forward, init_weights

__all__ = [
    'SpikingVGG', 'MultiStepSpikingVGG',
    'multi_step_spiking_vgg11','spiking_vgg11',
    'multi_step_spiking_vgg13','spiking_vgg13',
    'multi_step_spiking_vgg16','spiking_vgg16',
    'multi_step_spiking_vgg19','spiking_vgg19',
    'multi_step_spiking_vgg_custom','spiking_vgg_custom',
]

# modified from https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py




def get_model(args):
    norm_layer = nn.BatchNorm2d if args.bn else None
    ms_neuron = neuron.MultiStepParametricLIFNode

    family, version = args.model.split('-')
    if family == "vgg":
        if version == "custom":
                return multi_step_spiking_vgg_custom(
                    2*args.tbin, cfg=args.cfg,
                    norm_layer=norm_layer, multi_step_neuron=ms_neuron,
                    num_classes=2, backend="torch"
                )
        else:
            vggs = {
                "11": multi_step_spiking_vgg11,
                "13": multi_step_spiking_vgg13,
                "16": multi_step_spiking_vgg16,
            }
            return vggs[version](
                2*args.tbin, norm_layer=norm_layer,
                multi_step_neuron=ms_neuron,
                num_classes=2, backend="torch"
            )

class DetectionBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
       
        self.model = get_model(args)
            
        self.out_channels = self.model.out_channels
        extras_fm = args.extras
        
        self.extras = nn.ModuleList(
            [
                nn.Sequential(
                    SpikingBlock(self.out_channels[-1], self.out_channels[-1]//2, kernel_size=1),
                    SpikingBlock(self.out_channels[-1]//2, extras_fm[0], kernel_size=3, padding=1, stride=2),
                ),
                nn.Sequential(
                    SpikingBlock(extras_fm[0], extras_fm[0]//4, kernel_size=1),
                    SpikingBlock(extras_fm[0]//4, extras_fm[1], kernel_size=3, padding=1, stride=2),
                ),
                nn.Sequential(
                    SpikingBlock(extras_fm[1], extras_fm[1]//2, kernel_size=1),
                    SpikingBlock(extras_fm[1]//2, extras_fm[2], kernel_size=3, padding=1, stride=2),
                ),
            ]
        )

        self.extras.apply(init_weights)
        self.out_channels.extend(extras_fm)
    
    def forward(self, x):
        feature_maps = self.model(x, classify=False)
        x = feature_maps[-1]
        detection_feed = [time_to_batch(fm)[0] for fm in feature_maps]

        for block in self.extras:
            x = block(x)
            detection_feed.append(time_to_batch(x)[0])
            
        return detection_feed


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