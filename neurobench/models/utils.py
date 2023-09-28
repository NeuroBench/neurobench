from torch import nn
import snntorch as snn


def activation_modules():
    """
    The activation layers that can be auto-deteced. Every activation layer can only be included once.
    """
    return list(set([nn.ReLU,
            nn.Sigmoid,
           ]))
