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

spike_grad = surrogate.fast_sigmoid() #slope=25)
beta = 0.9
alpha = 0.95
warmup_steps = 10


def remove_sequential(network, all_layers):

    for name, layer in network.named_children():
        if isinstance(layer, nn.Sequential): # if sequential layer, apply recursively to layers in sequential layer
            #print(layer)
            remove_sequential(layer, all_layers)
        else: # if leaf node, add it to list
            # print(layer)
            all_layers.append((name,layer))


class SNN(nn.Module):
    def __init__(self,load_model=None, hidden_size=256, drop=False, rec=True, ns_readout=False, latent_layer_num=100):
        """Taken from: https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html"""
        super().__init__()


        self.hidden_size = hidden_size
        self.rec = rec


        self.forward1 = nn.Linear(20, hidden_size)
        self.rec1 = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.neuron1 = snn.Synaptic(alpha=torch.full((hidden_size,),alpha), learn_alpha=True, beta=torch.full((hidden_size,),beta), spike_grad=spike_grad, learn_beta=True, init_hidden=True)
        self.neuron1 = snn.Synaptic(alpha=alpha, beta=torch.full((hidden_size,),beta), spike_grad=spike_grad, learn_beta=True, init_hidden=True)
        self.forward2 = nn.Linear(hidden_size, hidden_size)
        self.rec2 = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.neuron2 = snn.Synaptic(alpha=torch.full((hidden_size,),alpha), learn_alpha=True, beta=torch.full((hidden_size,),beta), spike_grad=spike_grad, learn_beta=True, init_hidden=True)
        self.neuron2 = snn.Synaptic(alpha=alpha, beta=torch.full((hidden_size,),beta), spike_grad=spike_grad, learn_beta=True, init_hidden=True)

        nn.init.orthogonal_(self.rec1.weight)
        nn.init.orthogonal_(self.rec2.weight)

        self.net = nn.Sequential(
            self.forward1,
            self.neuron1,
            self.forward2,
            self.neuron2
        )


        self.output = nn.Linear(hidden_size, 200, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.ns_readout = ns_readout
        if self.ns_readout:
            self.alpha = nn.Parameter(torch.full((200,),0.9))
        else:
            self.output_neurons = snn.Synaptic(alpha=torch.full((200,),alpha), learn_alpha=True, beta=torch.full((200,),beta), spike_grad=spike_grad, learn_beta=True, init_hidden=True, output=True)


    def forward(self, x, latent_input=None, return_lat_acts=False):

        utils.reset(self.neuron1)
        utils.reset(self.neuron2)
        utils.reset(self.net)
        if self.ns_readout:
            out = torch.zeros(x.size(0), 200).to(x.get_device())
        else:
            utils.reset(self.output_neurons)

        softmax_output = torch.zeros(x.size(0), 200).to(x.get_device())
        num_steps = x.shape[1]

        # spk_rec = []
        spk1 = torch.zeros(x.size(0), self.hidden_size).to(x.get_device())
        spk2 = torch.zeros(x.size(0), self.hidden_size).to(x.get_device())
        spk1_list = []
        spk2_list = []
        

        # for step in range(num_steps):

        #     act1 = self.forward1(x[:,step])
        #     spk1 = self.neuron1(act1)

        #     act2 = self.forward2(spk1)
        #     spk2 = self.neuron2(act2)           

        #     outputs = self.output(spk2)

        #     if self.ns_readout:
        #         out = self.alpha*out + (1-self.alpha)* outputs
        #     else:
        #         spk_out, _, out = self.output_neurons(outputs)

        Wx1 = self.forward1(x)

        for step in range(num_steps):

            if self.rec:
                Vrec1 = self.rec1.weight.clone().fill_diagonal_(0)
                act1 = Wx1[:,step] + torch.matmul(spk1, Vrec1)
            else:
                act1 = Wx1[:,step]

            spk1 = self.neuron1(act1)
            spk1_list.append(spk1)
        spk1_stack = torch.stack(spk1_list, dim=1)
        
        Wx2 = self.forward2(spk1_stack)

        for step in range(num_steps):

            
            if self.rec:
                Vrec2 = self.rec2.weight.clone().fill_diagonal_(0)
                act2 = Wx2[:,step] + torch.matmul(spk2, Vrec2)
            else:
                act2 = Wx2[:,step]

            spk2 = self.neuron2(act2)
            spk2_list.append(spk2)
        spk2_stack = torch.stack(spk2_list, dim=1)


        outputs = self.output(spk2_stack)

        for step in range(num_steps):

            if self.ns_readout:
                out = self.alpha*out + (1-self.alpha)* outputs[:,step]
            else:
                spk_out, _, out = self.output_neurons(outputs[:,step])
                # spk_rec.append(spk_out)

            if step > warmup_steps :
                softmax_output += self.softmax(out)

        if return_lat_acts:
            return torch.stack(spk_rec), orig_acts
        else:
            return softmax_output
            # return torch.stack(spk_rec)