import torch.nn as nn
import torch.nn.functional as F
from torch import cat, mul
import collections
import torch

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils

spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.9
warmup_steps = 50


def remove_sequential(network, all_layers):

    for name, layer in network.named_children():
        if isinstance(layer, nn.Sequential): # if sequential layer, apply recursively to layers in sequential layer
            #print(layer)
            remove_sequential(layer, all_layers)
        else: # if leaf node, add it to list
            # print(layer)
            all_layers.append((name,layer))


class SNN(nn.Module):
    def __init__(self,load_model=None, drop=False, latent_layer_num=100):
        """Taken from: https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html"""
        super().__init__()

        if load_model:
            all_layers = []
            # all_seq = nn.Sequential(load_model.lat_features, load_model.end_features)
            remove_sequential(load_model.lat_features, all_layers)

        else:
            net = nn.Sequential(
                # nn.Flatten(),
                nn.Linear(20, 256),
                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                nn.Linear(256, 256),
                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                nn.Linear(256, 256),
                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                nn.Linear(256, 256),
                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
            )
            all_layers = []
            remove_sequential(net, all_layers)

        lat_list = []
        end_list = []

        for i, layer in enumerate(all_layers):
            if i <= latent_layer_num:
                lat_list.append(layer)
            else:
                end_list.append(layer)

        self.lat_features = nn.Sequential(collections.OrderedDict(lat_list))
        self.end_features = nn.Sequential(collections.OrderedDict(end_list))

        self.output = nn.Linear(256, 200, bias=False)
        self.output_neurons = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, latent_input=None, return_lat_acts=False):

        utils.reset(self.lat_features)
        utils.reset(self.end_features)
        utils.reset(self.output_neurons)
        softmax_output = torch.zeros(x.size(0), 200).to(x.get_device())
        # mem1 = self.lif1.init_leaky()
        # mem2 = self.lif2.init_leaky()
        # mem3 = self.lif3.init_leaky()
        # mem4 = self.lif4.init_leaky()
        # mem = []

        num_steps = x.shape[1]

        for step in range(num_steps):

            orig_acts = self.lat_features(x[:,step])
            if latent_input is not None:
                lat_acts = cat((orig_acts, latent_input), 0)
            else:
                lat_acts = orig_acts

            logits = self.end_features(lat_acts)
            # logits = F.avg_pool1d(logits, logits.shape[-1])
            # logits = logits.permute(0, 2, 1)

            outputs = self.output(logits)
            _, mem_out = self.output_neurons(outputs)

            if step > warmup_steps :
                softmax_output += self.softmax(mem_out)

        if return_lat_acts:
            return softmax_output, orig_acts
        else:
            return softmax_output