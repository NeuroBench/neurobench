import torch
from torch import Tensor
from neurobench.metrics.utils.decorators import check_shapes

from neurobench.metrics.abstract.workload_metric import WorkloadMetric


class MSE(WorkloadMetric):
    """Mean squared error of the model predictions."""

    def __init__(self):
        """Initialize the MSE metric."""

        super().__init__(requires_hooks=False)

    @check_shapes
    def __call__(self, model, preds: Tensor, data: Tensor) -> float:
        """
        Compute mean squared error.

        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            float: Mean squared error.

        """

        return torch.mean((preds - data[1]) ** 2).item()
