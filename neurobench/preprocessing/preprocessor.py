class NeuroBenchPreProcessor:
    """
    Abstract class for NeuroBench pre-processors.

    Individual pre-processors are responsible for implementing init and call functions.

    """

    def __init__(self, args):
        """
        Initialize pre-processor with any parameters needed.

        Args:
            args: Any arguments needed for pre-processing.

        """
        raise NotImplementedError(
            "Subclasses of NeuroBenchPreProcessor should implement __init__"
        )

    def __call__(self, dataset):
        """
        Process dataset of format (data, targets), or (data, targets, kwargs) to prepare
        for model inference.

        Args:
            dataset: A tuple of (data, targets) or (data, targets, kwargs) where data is a PyTorch tensor of shape (batch, timesteps, ...)

        """
        raise NotImplementedError(
            "Subclasses of NeuroBenchPreProcessor should implement __call__"
        )
