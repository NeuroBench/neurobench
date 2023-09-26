from torch import nn
import torch
import snntorch as snn
from neurobench.benchmarks.hooks import ActivationHook

from neurobench.benchmarks.utils.metric_utils import activation_modules


class NeuroBenchModel:
    """ Abstract class for NeuroBench models. Individual model frameworks are
    responsible for defining model inference.
    """

    def __init__(self, net):
        """ Init using a trained network
        
        Args:
            net: A trained network
        """
        self.activation_modules = activation_modules()
        self.activation_hooks = []
        self.connection_hooks = []
        self.first_layer = None


    def __call__(self, batch):
        """ Includes the whole pipeline from data to inference (output should be same format as targets).

        Args:
            batch: A batch of data to run inference on
        """
        raise NotImplementedError("Subclasses of NeuroBenchModel should implement __call__")

    def __net__(self):
        """ Returns the underlying network
        """
        raise NotImplementedError("Subclasses of NeuroBenchModel should implement __net__")

    def set_first_layer(self, layer):
        """ Sets the first layer of the network
        """
        self.first_layer = layer
        
    def add_activation_module(self, activaton_module):
        """ Add a cutomized activaton_module that can be detected after running the preprocessing pipeline for detecting activation functions
        """
        self.activation_modules.append(activaton_module)

    def activation_layers(self):
        """ Retrieve all the activaton layers of the underlying network (including spiking neuron layers)
        """

        def get_activation_layers(parent):
            """ Returns all the neuro layers
            """
            act_layers = []
            children = parent.children()
            for child in children:
                grand_children = list(child.children())
                if len(grand_children) == 0:  # leaf child
                    if isinstance(child, snn.SpikingNeuron):
                        act_layers.append(child)
                    else:
                        for activaton_module in self.activation_modules:
                            # add all the activation act_layers and spiking neuron act_layers
                            if isinstance(child, activaton_module):
                                act_layers.append(child)

                else:
                    children_layers = get_activation_layers(child)
                    act_layers.extend(children_layers)
            
            return act_layers
        
        root = self.__net__()
        act_layers = get_activation_layers(root)
        return act_layers

    def connection_layers(self):
        """ Retrieve all the connection layers of the underlying network (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)
        """
        supported_layers = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)

        def get_connection_layers(parent):
            """ Returns all the connection layers
            """
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
        """ Resets all the hooks (activation hooks and connection hooks)
        """
        for hook in self.activation_hooks:
            hook.reset()

        for hook in self.connection_hooks:
            hook.reset()