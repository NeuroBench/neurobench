import torch
from neurobench.metrics.abstract.workload_metric import AccumulatedMetric
from neurobench.metrics.utils.decorators import check_shapes


class R2(AccumulatedMetric):
    """
    R2 Score of the model predictions.

    Currently implemented for 2D output only.

    """

    def __init__(self):
        """
        Initalize metric state.

        Must hold memory of all labels seen so far.

        """
        super().__init__(requires_hooks=False)
        self.x_sum_squares = 0.0
        self.y_sum_squares = 0.0

        self.x_labels = None
        self.y_labels = None

    def reset(self):
        """Reset metric state."""
        self.x_sum_squares = 0.0
        self.y_sum_squares = 0.0

        self.x_labels = torch.tensor([])
        self.y_labels = torch.tensor([])

    @check_shapes
    def __call__(self, model, preds, data):
        """
        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            float: R2 Score.
        """
        self.x_sum_squares += torch.sum((data[1][:, 0] - preds[:, 0]) ** 2).item()
        self.y_sum_squares += torch.sum((data[1][:, 1] - preds[:, 1]) ** 2).item()

        if self.x_labels is None:
            self.x_labels = data[1][:, 0]
            self.y_labels = data[1][:, 1]
        else:
            self.x_labels = torch.cat((self.x_labels, data[1][:, 0]))
            self.y_labels = torch.cat((self.y_labels, data[1][:, 1]))

        return self.compute()

    def compute(self):
        """Compute r2 score using accumulated data."""
        x_denom = self.x_labels.var(correction=0) * len(self.x_labels)
        y_denom = self.y_labels.var(correction=0) * len(self.y_labels)

        x_r2 = 1 - (self.x_sum_squares / x_denom)
        y_r2 = 1 - (self.y_sum_squares / y_denom)

        r2 = (x_r2 + y_r2) / 2

        return r2.item()
