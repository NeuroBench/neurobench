from abc import ABC, abstractmethod
from neurobench.hooks.neuron import NeuronHook
from neurobench.hooks.layer import LayerHook
from neurobench.blocks.activation import SUPPORTED_ACTIVATIONS
from neurobench.blocks.layer import SUPPORTED_LAYERS


class NeuroBenchModel(ABC):
    """
    Abstract class for NeuroBench models.

    Individual model frameworks are responsible for defining model inference.

    """

    def __init__(self):
        """
        Init using a trained network.

        Args:
            net: A trained network

        """
        self.activation_modules = list(SUPPORTED_ACTIVATIONS)
        self.activation_hooks = []
        self.connection_hooks = []
        self.first_layer = None

        # self.supported_layers = (
        #     nn.Linear,
        #     nn.Conv2d,
        #     nn.Conv1d,
        #     nn.Conv3d,
        #     nn.RNNBase,
        #     nn.RNNCellBase,
        # )

    @abstractmethod
    def __call__(self, batch):
        """
        Includes the whole pipeline from data to inference (output should be same format
        as targets).

        Args:
            batch: A batch of data to run inference on

        """
        pass

    @abstractmethod
    def __net__(self):
        """Returns the underlying network."""
        pass

    def set_first_layer(self, layer):
        """Sets the first layer of the network."""
        self.first_layer = layer

    def add_activation_module(self, activaton_module):
        """Add a cutomized activaton_module that can be detected after running the
        preprocessing pipeline for detecting activation functions."""
        self.activation_modules.append(activaton_module)

    def activation_layers(self):
        """
        Retrieve all activation layers in the network, including spiking neurons.

        Returns:
            list: Activation layers.

        """

        def is_activation_layer(module):
            """Check if a module is an activation layer."""
            return any(
                isinstance(module, act_mod) for act_mod in self.activation_modules
            )

        def find_activation_layers(module):
            """Recursively find activation layers in a module."""
            layers = []
            for child_name, child in module.named_children():
                if is_activation_layer(child):
                    layers.append({"layer_name": child_name, "layer": child})
                elif list(child.children()):  # Check for nested submodules
                    layers.extend(find_activation_layers(child))
            return layers

        return find_activation_layers(self.__net__())

    def connection_layers(self):
        """
        Retrieve all connection layers in the network.

        Connection layers include Linear, Conv, and RNN-based layers.

        Returns:
            list: Connection layers.

        """

        def find_connection_layers(module):
            """Recursively find connection layers in a module."""
            layers = []
            for child in module.children():
                if isinstance(child, SUPPORTED_LAYERS):
                    layers.append(child)
                elif list(child.children()):  # Check for nested submodules
                    layers.extend(find_connection_layers(child))
            return layers

        return find_connection_layers(self.__net__())

    def reset_hooks(self):
        """Resets all the hooks (activation hooks and connection hooks) in parallel."""
        for hook in self.activation_hooks + self.connection_hooks:
            hook.reset()

    def close_hooks(self):
        """Closes all the hooks (activation hooks and connection hooks)"""
        for hook in self.activation_hooks + self.connection_hooks:
            hook.close()

    def cleanup_hooks(self):
        """Closes all the hooks (activation hooks and connection hooks)"""
        for hook in self.activation_hooks + self.connection_hooks:
            hook.reset()
            hook.close()

        self.activation_hooks.clear()
        self.connection_hooks.clear()

    def register_hooks(self):
        """Registers hooks for the model."""

        # Registered activation hooks
        for layer in self.activation_layers():
            layer_name = layer["layer_name"]
            layer = layer["layer"]
            self.activation_hooks.append(NeuronHook(layer, layer_name))

        for layer in self.connection_layers():
            self.connection_hooks.append(LayerHook(layer))
