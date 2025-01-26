from abc import ABC, abstractmethod
from torch import Tensor


class NeuroBenchPostProcessor(ABC):
    """
    Abstract class for NeuroBench postprocessors.

    Postprocessors take the spiking output from the models and provide several methods
    of combining them. Individual postprocessors are responsible for implementing init
    and call functions.

    """

    @abstractmethod
    def __call__(self, spikes: Tensor) -> Tensor:
        """
        Process tensor of spiking data of format (batch, timesteps, ...) to match spikes
        to ground truth.

        Args:
            spikes (Tensor): A torch tensor of spikes output by a NeuroBenchModel of
                shape (batch, timestep, ...)
        Returns:
            Tensor: A tensor of shape (batch, ...) with the processed spikes

        """
