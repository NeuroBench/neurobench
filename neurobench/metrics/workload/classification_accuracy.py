import torch
from torch import Tensor
from neurobench.metrics.utils.decorators import check_shapes

from neurobench.metrics.abstract.workload_metric import WorkloadMetric


class ClassificationAccuracy(WorkloadMetric):
    """Classification accuracy of the model predictions."""

    def __init__(self):
        """Initialize the ClassificationAccuracy metric."""
        super().__init__(requires_hooks=False)

    @check_shapes
    def __call__(self, model, preds, data):
        """
        Compute classification accuracy.

        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            float: Classification accuracy

        """

        equal = torch.eq(preds, data[1])
        return torch.mean(equal.float()).item()
