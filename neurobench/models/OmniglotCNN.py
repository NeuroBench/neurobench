import torch.nn as nn

def conv_block(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class OmniglotCNN(nn.Sequential):
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64, num_ways = 5):
        super(OmniglotCNN, self).__init__(*[
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            nn.Linear(z_dim, num_ways)
        ])