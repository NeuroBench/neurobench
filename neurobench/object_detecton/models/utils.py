import torch.nn as nn

from spikingjelly.clock_driven import layer, neuron, surrogate
from .spiking_densenet import *
from .spiking_squeezenet import *
from .spiking_mobilenet import *
from .spiking_vgg import *

class SpikingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                   bias=False, stride=1, padding=0, groups=1, backend='cupy'):
        super().__init__()
        
        self.bn_conv = layer.SeqToANNContainer(
            nn.ConstantPad2d(padding, 0.),
		    nn.BatchNorm2d(in_channels),
		    nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, 
			      stride=stride, padding=0, groups=groups),
		)
        
        self.neuron = neuron.MultiStepParametricLIFNode(
            init_tau=2.0, v_threshold=1., 
            surrogate_function=surrogate.ATan(),
            detach_reset=True, backend=backend,
        )

    def forward(self, x):
        out = self.bn_conv(x)
        out = self.neuron(out)
        return out


def get_model(args):
    norm_layer = nn.BatchNorm2d if args.bn else None
    ms_neuron = neuron.MultiStepParametricLIFNode

    family, version = args.model.split('-')
    if family == "densenet":
        depth, growth_rate = version.split('_')
        blocks = {"121": [6,12,24,16], "169": [6,12,32,32]}
        return multi_step_spiking_densenet_custom(
            2*args.tbin, norm_layer=norm_layer,
            multi_step_neuron=ms_neuron,
            growth_rate=int(growth_rate), block_config=blocks[depth],
            num_classes=2, backend="cupy",
        )
    elif family == "mobilenet":
        return multi_step_spiking_mobilenet(
            2*args.tbin, norm_layer=norm_layer,
            in_channels=int(version),
            multi_step_neuron=ms_neuron,
            num_classes=2, dw=True, backend="cupy", 
        )
    elif family == "squeezenet":
        squeezenets = {
            "1.0": multi_step_spiking_squeezenet1_0,
            "1.1": multi_step_spiking_squeezenet1_1,
        }
        return squeezenets[version](
            2*args.tbin, norm_layer=norm_layer,
            multi_step_neuron=ms_neuron,
            num_classes=2, backend="cupy"
        )
    elif family == "vgg":
        if version == "custom":
                return multi_step_spiking_vgg_custom(
                    2*args.tbin, cfg=args.cfg,
                    norm_layer=norm_layer, multi_step_neuron=ms_neuron,
                    num_classes=2, backend="cupy"
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
                num_classes=2, backend="cupy"
            )