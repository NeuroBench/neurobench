from torch import nn
import snntorch as snn
from neurobench.benchmarks.hooks import ActivationHook

from .utils import activation_modules


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
        # for layer in self.activation_layers():
        #     self.activation_hooks.append(ActivationHook(layer))


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

    def add_activation_module(self, activaton_module):
        """ Add a cutomized activaton_module that can be detected after running the preprocessing pipeline for detecting activation functions
        """
        self.activation_modules.append(activaton_module)

    def activation_layers(self):
        """ Retrieve all the activaton layers of the underlying network
        """
        def get_activation_layers(parent):
            """ Returns all the neuro layers
            """
            layers = []

            children = parent.children()
            for child in children:
                grand_children = list(child.children())
                if len(grand_children) == 0:  # leaf child
                    if isinstance(child, snn.SpikingNeuron):
                        layers.append(child)

                    else:
                        for activaton_module in self.activation_modules:
                            # add all the activation layers and spiking neuron layers
                            if isinstance(child, activaton_module):
                                layers.append(child)
                else:
                    get_activation_layers(child)
                    # print
                    # layers.extend(children_layers)
            
            return layers
        
        root = self.__net__()
        layers = get_activation_layers(root)
        return layers
'''
            children = list(module.children())
            
            if len(children) == 0: # it is a leaf
                for activaton_module in self.activation_modules:
                    # add all the activation layers and spiking neuron layers
                    if isinstance(module, activaton_module) or isinstance(module, snn.SpikingNeuron):
                        return module
            
            else:
                layers = []
                for child in children:
                    layers.append(get_activation_layers(child))
                return layers
        # Pull the layers from the model's network
        layers = self.__net__().children()
        print(self.__net__().children())
        # For each layer, count where the weights are zero
        layers_act = []
        for module in layers:
            print(module)
            layers_act.extend(get_activation_layers(module))


        # Return the ratio of zeros to weights, rounded to 3 decimals
        return layers_act
                    '''
