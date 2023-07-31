"""
=====================================================================
Project:      NeuroBench
File:         MackeyGlass-ESN_taus.py
Description:  Python code benchmarking on the Mackey-Glass task
Date:         20. July 2023
=====================================================================
Copyright stuff
=====================================================================
"""


import sys
sys.path.append("../../..")

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from neurobench.models.echo_state_network import EchoStateNetwork
from neurobench.datasets.mackey_glass import MackeyGlass

# TODO: reservoir should be added as a model parameter. also should experiment with FP precision

# TODO: should we only count parameters for num_params, model_size if they are nonzero?

def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size(model, data, preds):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return param_size + buffer_size

# calculate model connection sparsity based off the Linear layers in the network
# TODO: should this be called something other than connection sparsity? since it is reporting the ratio of actual to possible synapses
def model_connection_sparsity(model):
    possible_synapses = 0
    actual_synapses = 0

    for module in model.modules():
        if isinstance(module, nn.Linear):
            possible_synapses += torch.numel(module.weight.data)
            actual_synapses += torch.count_nonzero(module.weight.data)

    return actual_synapses / possible_synapses
        
# Load ESN
esn = torch.load('esn.pth')

params = num_params(esn)
print("Params:", params)

sparsity = model_connection_sparsity(esn)
print("Sparsity:", sparsity)

size = model_size(esn, None, None)
print("Size:", size)





