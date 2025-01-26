from neurobench.metrics.abstract import StaticMetric


class ParameterCount(StaticMetric):
    """A metric that counts the number of parameters in a model."""

    def __call__(self, model):
        """
        Count the number of parameters in a model.

        Args:
            model: A NeuroBenchModel.
        Returns:
            int: Number of parameters in the model.

        """
        return sum(p.numel() for p in model.__net__().parameters())
