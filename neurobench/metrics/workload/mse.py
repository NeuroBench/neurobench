import torch
from torch import Tensor
from neurobench.metrics.utils.decorators import check_shapes

from neurobench.metrics.abstract.workload_metric import WorkloadMetric


class MSE(WorkloadMetric):

    @check_shapes
    def __call__(self, model, preds: Tensor, data: Tensor) -> float:
        """
        Mean squared error of the model predictions.

        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            float: Mean squared error.

        """

        return torch.mean((preds - data[1]) ** 2).item()
