from torch import nn


class NeuroBenchModel:
    """ Abstract class for NeuroBench models. Individual model frameworks are
    responsible for defining model inference.
    """

    def __init__(self, net):
        """ Init using a trained network
        
        Args:
            net: A trained network
        """
        raise NotImplementedError("Subclasses of NeuroBenchModel should implement __init__")

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
    
    def __neuro_layers__(self):
        """ Returns all the neuro layers
        """
        raise NotImplementedError("Subclasses of NeuroBenchModel should implement __neuro_layers__")


class NeuroBenchNetwork(nn.Module):
    """ Abstract class for the trained network in NeuroBenchModel.
    """
    def __neuro_layers__(self):
        """ Returns all the neuro layers
        """
        raise NotImplementedError("Subclasses of NeuroBenchNetwork should implement __neuro_layers__")
