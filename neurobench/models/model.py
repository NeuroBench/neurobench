"""
"""

class NeuroBenchModel:
    """
    Abstract class for NeuroBench models. Individual model frameworks are
    responsible for defining model inference.
    """

    def __init__(self, net):
        """
        Init using a trained network.
        """
        raise NotImplementedError("Subclasses of NeuroBenchModel should implement __init__")

    def __call__(self, batch):
        """
        Includes the whole pipeline from data to inference (output should be same format as targets).
        """
        raise NotImplementedError("Subclasses of NeuroBenchModel should implement __call__")
