import torch.nn as nn
import torch.nn.functional as F
from torch import cat, mul
import collections

def remove_sequential(network, all_layers):

    for name, layer in network.named_children():
        if isinstance(layer, nn.Sequential): # if sequential layer, apply recursively to layers in sequential layer
            #print(layer)
            remove_sequential(layer, all_layers)
        else: # if leaf node, add it to list
            # print(layer)
            all_layers.append((name,layer))

class M5_Feature(nn.Module):
    def __init__(self, n_input=1, stride=16, n_channel=32, input_kernel=80, pool_kernel=4):

        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=input_kernel, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(pool_kernel)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(pool_kernel)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(pool_kernel)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(pool_kernel)



    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.act3(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.act4(self.bn4(x))
        x = self.pool4(x)

        return x

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, 
                 n_channel=32, load_model=None, output= None, input_kernel=80, pool_kernel=4, latent_layer_num=100):
        """Taken from: https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html"""
        super().__init__()

        if load_model:
            all_layers = []
            # all_seq = nn.Sequential(load_model.lat_features, load_model.end_features)
            remove_sequential(load_model.lat_features, all_layers)

        else:
            features = M5_Feature(n_input, stride, n_channel, input_kernel, pool_kernel)
            
            all_layers = []
            remove_sequential(features, all_layers)

        lat_list = []
        end_list = []

        for i, layer in enumerate(all_layers):
            if i <= latent_layer_num:
                lat_list.append(layer)
            else:
                end_list.append(layer)

        self.lat_features = nn.Sequential(collections.OrderedDict(lat_list))
        self.end_features = nn.Sequential(collections.OrderedDict(end_list))

        if load_model:
            self.output = load_model.output
        else:
            self.output = nn.Linear(2 * n_channel, n_output, bias=False)

    def forward(self, x, latent_input=None, return_lat_acts=False):

        orig_acts = self.lat_features(x)
        if latent_input is not None:
            lat_acts = cat((orig_acts, latent_input), 0)
        else:
            lat_acts = orig_acts

        logits = self.end_features(lat_acts)
        logits = F.avg_pool1d(logits, logits.shape[-1])
        logits = logits.permute(0, 2, 1)

        outputs = self.output(logits)

        # masked_outputs = torch.mul(self.mask, outputs)

        if return_lat_acts:
            return outputs, orig_acts
        else:
            return outputs

