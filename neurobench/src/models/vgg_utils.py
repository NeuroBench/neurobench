from collections import OrderedDict

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, neuron, layer, surrogate




def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class SpikingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                   bias=False, stride=1, padding=0, groups=1, backend='torch'):
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
