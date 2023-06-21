"""
"""
import torch.nn as nn
import numpy as np


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

    def __init__(self, input_dim=96, layer1=32, layer2=48, output_dim=2, dropout_rate=0.5, hyperparams=None):
        super().__init__()
        self.input_size = input_dim
        self.hidden1 = layer1
        self.hidden2 = layer2
        self.output_size = output_dim

        self.fc1 = nn.Linear(self.input_size, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.fc3 = nn.Linear(self.hidden2, self.output_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm1 = nn.BatchNorm1d(self.hidden1)
        self.batchnorm2 = nn.BatchNorm1d(self.hidden2)
        self.hyperparams = hyperparams

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.batchnorm1(x)
        x = self.activation(self.dropout(self.fc2(x)))
        x = self.batchnorm2(x)
        x = self.fc3(x)

        return x, 1

    def get_dimensions(self):
        return np.array([self.input_size, self.hidden1, self.hidden2, self.output_size])

    def get_recurrent_layers(self):
        return np.array([0, 0, 0])

    def get_decaying_variables(self):
        return np.array([0, 0, 0])

    def get_bias(self):
        return np.array([0, 0, 0])

    def get_bn(self):
        return np.array([1, 1, 0])

    def get_algorithmic_timestep(self):
        return .016

    def get_binning_window(self):
        return self.hyperparams['window']
