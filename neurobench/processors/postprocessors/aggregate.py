from neurobench.processors.abstract.postprocessor import NeuroBenchPostProcessor


class Aggregate(NeuroBenchPostProcessor):
    """Returns aggregated spikes."""

    def __call__(self, spikes):
        """
        Returns the aggregated spikes.

        Args:
            spikes (Tensor): A torch tensor of spikes of shape (batch, timestep, classes)

        Returns:
            Tensor: A torch tensor of spikes of shape (batch, classes)

        """
        return spikes.sum(1)
