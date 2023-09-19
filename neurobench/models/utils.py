from torch import nn
import snntorch as snn


def activation_modules():
    """
    The activation layers that can be auto-deteced.
    """
    return [nn.ReLU,
            nn.Sigmoid,

           ]
