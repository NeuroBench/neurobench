"""
"""

class NeuroBenchProcessor():
    """
    Abstract class for NeuroBench pre-processors. Individual pre-processors are
    responsible for implementing init and call functions.
    """

    def __init__(self, args):
        """
        Initialize pre-processor with any parameters needed
        """
        raise NotImplementedError("Subclasses of NeuroBenchProcessor should implement __init__")

    def __call__(self, dataset):
        """
        Process dataset of format (data, targets) to prepare for model inference
        """
        raise NotImplementedError("Subclasses of NeuroBenchProcessor should implement __call__")
