import torch.nn as nn
import torch.nn.functional as F
from torch import cat, mul

def remove_sequential(network, all_layers):

    for layer in network.children():
        if isinstance(layer, nn.Sequential): # if sequential layer, apply recursively to layers in sequential layer
            #print(layer)
            remove_sequential(layer, all_layers)
        else: # if leaf node, add it to list
            # print(layer)
            all_layers.append(layer)

class M5_Feature(nn.Module):
    def __init__(self, n_input=1, stride=16, n_channel=32):

        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)

        return x

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, 
                 n_channel=32, seq_model=None, output= None, latent_layer_num=100):
        """Taken from: https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html"""
        super().__init__()

        if seq_model:
            all_layers = []
            remove_sequential(seq_model, all_layers)

            self.output = output # all_layers.pop(-1)
            # all_layers = all_layers[:-1]

        else:
            self.features = M5_Feature(n_input, stride, n_channel)
            self.fc1 = nn.Linear(2 * n_channel, n_output)
            all_layers = []
            remove_sequential(self.features, all_layers)

        lat_list = []
        end_list = []

        for i, layer in enumerate(all_layers):
            if i <= latent_layer_num:
                lat_list.append(layer)
            else:
                end_list.append(layer)

        self.lat_features = nn.Sequential(*lat_list)
        self.end_features = nn.Sequential(*end_list)

    def forward(self, x, latent_input=None, return_lat_acts=False):

        orig_acts = self.lat_features(x)
        if latent_input is not None:
            lat_acts = cat((orig_acts, latent_input), 0)
        else:
            lat_acts = orig_acts

        logits = self.end_features(lat_acts)

    
        outputs = self.output(logits.squeeze())

        # masked_outputs = torch.mul(self.mask, outputs)

        if return_lat_acts:
            return outputs, orig_acts
        else:
            return outputs

