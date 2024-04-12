from torch import nn
import torch
from neurobench.benchmarks.hooks import ActivationHook

from neurobench.utils import activation_modules


class NeuroBenchModel:
    """
    Abstract class for NeuroBench models.

    Individual model frameworks are responsible for defining model inference.

    """

    def __init__(self, net):
        """
        Init using a trained network.

        Args:
            net: A trained network

        """
        self.activation_modules = activation_modules()
        self.activation_hooks = []
        self.connection_hooks = []
        self.first_layer = None

        self.supported_layers = (
            torch.nn.Linear,
            torch.nn.Conv2d,
            torch.nn.Conv1d,
            torch.nn.Conv3d,
            torch.nn.RNNBase,
            torch.nn.RNNCellBase,
        )

    def __call__(self, batch):
        """
        Includes the whole pipeline from data to inference (output should be same format
        as targets).

        Args:
            batch: A batch of data to run inference on

        """
        raise NotImplementedError(
            "Subclasses of NeuroBenchModel should implement __call__"
        )

    def __net__(self):
        """Returns the underlying network."""
        raise NotImplementedError(
            "Subclasses of NeuroBenchModel should implement __net__"
        )

    def set_first_layer(self, layer):
        """Sets the first layer of the network."""
        self.first_layer = layer

    def add_activation_module(self, activaton_module):
        """Add a cutomized activaton_module that can be detected after running the
        preprocessing pipeline for detecting activation functions."""
        self.activation_modules.append(activaton_module)

    def activation_layers(self):
        """Retrieve all the activaton layers of the underlying network (including
        spiking neuron layers)"""

        def check_if_activation(module):
            for activation_module in self.activation_modules:
                if isinstance(module, activation_module):
                    return True

        def get_activation_layers(parent):
            """Returns all the neuro layers."""
            layers = []
            children = parent.children()
            for child in children:
                if check_if_activation(child):
                    # is an activation module
                    layers.append(child)
                else:
                    if len(list(child.children())) != 0:
                        # not an activation module and has nested submodules
                        children_layers = get_activation_layers(child)
                        layers.extend(children_layers)

            return layers

        root = self.__net__()
        layers = get_activation_layers(root)
        return layers

    def connection_layers(self):
        """Retrieve all the connection layers of the underlying network
        (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)"""
        supported_layers = self.supported_layers

        def get_connection_layers(parent):
            """Returns all the connection layers."""
            connection_layers = []
            children = parent.children()
            for child in children:
                grand_children = list(child.children())
                if len(grand_children) == 0:  # leaf child
                    if isinstance(child, supported_layers):
                        connection_layers.append(child)

                else:
                    children_layers = get_connection_layers(child)
                    connection_layers.extend(children_layers)

            return connection_layers

        root = self.__net__()
        connection_layers = get_connection_layers(root)
        return connection_layers

    def reset_hooks(self):
        """Resets all the hooks (activation hooks and connection hooks)"""
        for hook in self.activation_hooks:
            hook.reset()

        for hook in self.connection_hooks:
            hook.reset()
