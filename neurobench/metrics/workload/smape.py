import torch
from torch import Tensor
from neurobench.metrics.utils.decorators import check_shapes

from neurobench.metrics.abstract.workload_metric import WorkloadMetric


class SMAPE(WorkloadMetric):

    @check_shapes
    def __call__(self, model, preds: Tensor, data: Tensor) -> float:
        """
        Symmetric mean absolute percentage error of the model predictions.

        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            float: Symmetric mean absolute percentage error.

        """

        smape = 200 * torch.mean(
            torch.abs(preds - data[1]) / (torch.abs(preds) + torch.abs(data[1]))
        )
        return torch.nan_to_num(smape, nan=200.0).item()
