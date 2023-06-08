import torch.nn as nn

def conv_block(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class OmniglotCNN(nn.Sequential):
    def __init__(self, input_channels=1, hidden_channels=64, num_ways = 5):
        super(OmniglotCNN, self).__init__(*[
            conv_block(input_channels, hidden_channels),
            conv_block(hidden_channels, hidden_channels),
            conv_block(hidden_channels, hidden_channels),
            conv_block(hidden_channels, hidden_channels),
            nn.Linear(hidden_channels, num_ways)
        ])
