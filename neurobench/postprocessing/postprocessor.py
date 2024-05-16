import torch


class NeuroBenchPostProcessor:
    """
    Abstract class for NeuroBench postprocessors.

    Postprocessors take the spiking output from the models and provide several methods
    of combining them. Individual postprocessors are responsible for implementing init
    and call functions.

    """

    def __init__(self, args):
        """
        Initialize postprocessor with any parameters needed.

        Args:
            args: A dictionary of arguments for the postprocessor

        """
        raise NotImplementedError(
            "Subclasses of NeuroBenchPostProcessor should implement __init__"
        )

    def __call__(self, spikes):
        """
        Process tensor of spiking data of format (batch, timesteps, ...) to match spikes
        to ground truth.

        Args:
            spikes: A torch tensor of spikes output by a NeuroBenchModel of
                shape (batch, timestep, ...)

        """
        raise NotImplementedError(
            "Subclasses of NeuroBenchPostProcessor should implement __call__"
        )


def choose_max_count(spikes):
    """
    Returns the class with the highest spike count over the sample.

    Args:
        spikes: A torch tensor of spikes of shape (batch, timestep, classes)

    """
    # Sum across time and return index with highest count
    return spikes.sum(1).argmax(1)


def aggregate(spikes):
    """
    Returns the aggregated spikes.

    Args:
        spikes: A torch tensor of spikes of shape (batch, timestep, classes)

    Returns:
        spikes: A torch tensor of spikes of shape (batch, classes)

    """
    return spikes.sum(1)
