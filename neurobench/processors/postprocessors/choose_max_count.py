from neurobench.processors.abstract.postprocessor import NeuroBenchPostProcessor


class ChooseMaxCount(NeuroBenchPostProcessor):
    """Returns the class with the highest spike count over the sample."""

    def __call__(self, spikes):
        """
        Returns the class with the highest spike count over the sample.

        Args:
            spikes: A torch tensor of spikes of shape (batch, timestep, classes)

        """
        return spikes.sum(1).argmax(1)
