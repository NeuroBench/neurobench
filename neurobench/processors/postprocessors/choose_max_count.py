from neurobench.processors.abstract.postprocessor import NeuroBenchPostProcessor


class ChooseMaxCount(NeuroBenchPostProcessor):
    """Returns the class with the highest spike count over the sample."""

    def __call__(self, spikes):
        """
        Returns the class with the highest spike count over the sample.

        Args:
            spikes (Tensor): A torch tensor of spikes of shape (batch, timestep, classes)

        Returns:
            Tensor: A torch tensor of shape (batch,) with the class index of the highest spike count

        """
        return spikes.sum(1).argmax(1)
