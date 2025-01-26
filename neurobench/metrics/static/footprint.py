from neurobench.metrics.abstract import StaticMetric


class Footprint(StaticMetric):
    """A metric that counts the memory footprint of a model."""

    def __call__(self, model):
        """
        Count the memory footprint of a model.

        Args:
            model: A NeuroBenchModel.
        Returns:
            float: Memory footprint of the model.

        """
        param_size = 0
        for param in model.__net__().parameters():
            param_size += param.numel() * param.element_size()

        buffer_size = 0
        for buffer in model.__net__().buffers():
            buffer_size += buffer.numel() * buffer.element_size()

        return param_size + buffer_size
