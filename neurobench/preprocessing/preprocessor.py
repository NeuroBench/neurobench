class NeuroBenchProcessor():
    """ Abstract class for NeuroBench pre-processors. Individual pre-processors are
    responsible for implementing init and call functions.
    """

    def __init__(self, args):
        """ Initialize pre-processor with any parameters needed

        Args:
            args: Any arguments needed for pre-processing.
        """
        raise NotImplementedError("Subclasses of NeuroBenchProcessor should implement __init__")

    def __call__(self, dataset):
        """ Process dataset of format (data, targets) to prepare for model inference

        Args:
            dataset: A tuple of (data, targets) where data is a PyTorch tensor of shape (batch, timesteps, ...)
        """
        raise NotImplementedError("Subclasses of NeuroBenchProcessor should implement __call__")
