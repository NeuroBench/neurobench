from abc import ABC, abstractmethod


class NeuroBenchPostProcessor(ABC):
    """
    Abstract class for NeuroBench postprocessors.

    Postprocessors take the spiking output from the models and provide several methods
    of combining them. Individual postprocessors are responsible for implementing init
    and call functions.

    """

    @abstractmethod
    def __call__(self, spikes):
        """
        Process tensor of spiking data of format (batch, timesteps, ...) to match spikes
        to ground truth.

        Args:
            spikes: A torch tensor of spikes output by a NeuroBenchModel of
                shape (batch, timestep, ...)

        """