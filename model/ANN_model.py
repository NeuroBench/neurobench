import torch
import torch.nn as nn


class ANNModel(nn.Module):
    """
    A straightforward 3-layer fully-connected network for predicting x&y coordinate
    of the test subject.
    :param input_dim: input feature dimension
    :param layer1: Number of hidden neurons in the first layer
    :param layer2: Number of hidden neurons in the second layer
    :param output_dim: output feature dimension
    :param dropout_rate: Probability of dropout
    """
    def __init__(self, input_dim=128, layer1=32, layer2=48, output_dim=2, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm1 = nn.BatchNorm1d(layer1)
        self.batchnorm2 = nn.BatchNorm1d(layer2)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.batchnorm1(x)
        x = self.activation(self.dropout(self.fc2(x)))
        x = self.batchnorm2(x)
        x = self.fc3(x)

        return x

