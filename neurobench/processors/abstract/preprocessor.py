from abc import ABC, abstractmethod
from torch import Tensor


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
    def __call__(self, dataset: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """
        Process dataset of format (data, targets), or (data, targets, kwargs) to prepare
        for model inference.

        Args:
            dataset (tuple): A tuple of (data, targets) or (data, targets, kwargs) where data is a PyTorch tensor of shape (batch, timesteps, ...)

        Returns:
            tuple: A tuple of (data, targets) or (data, targets, kwargs) where data is a PyTorch tensor of shape (batch, timesteps, ...)

        """
