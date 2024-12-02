from abc import ABC, abstractmethod


class NeuroBenchPreProcessor(ABC):
    """
    Abstract class for NeuroBench pre-processors.

    Individual pre-processors are responsible for implementing init and call functions.

    """

    @abstractmethod
    def __init__(self, *args):
        """
        Initialize pre-processor with any parameters needed.

        Args:
            args: Any arguments needed for pre-processing.

        """

    @abstractmethod
    def __call__(self, dataset):
        """
        Process dataset of format (data, targets), or (data, targets, kwargs) to prepare
        for model inference.

        Args:
            dataset: A tuple of (data, targets) or (data, targets, kwargs) where data is a PyTorch tensor of shape (batch, timesteps, ...)

        """
