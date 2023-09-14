from torch import nn

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
        """Add a cutomized activaton_module
        """
        self.activation_modules.append(activaton_module)
    
    def activation_layers(self):
        """ Returns all the neuro layers
        """
        layers = []

        children_layers = self.__net__().children()
        for layer in children_layers:
            for activaton_module in self.activation_modules:
                if isinstance(layer, activaton_module):
                    layers.append(layer)
        
        return layers

