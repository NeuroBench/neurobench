from abc import ABC, abstractmethod
from torch import Tensor
from neurobench.models import NeuroBenchModel


class WorkloadMetric(ABC):
    """
    Abstract base class for workload metrics.

    A workload metric is designed to evaluate some aspect of a model's performance
    or behavior, typically during the inference phase, based on its predictions
    and input data. This class defines the basic interface for all workload metrics
    that require computation over batches of data.

    Attributes:
        requires_hooks (bool): Flag indicating if the metric requires hooks for its computation.

    """

    def __init__(self, requires_hooks: bool = False):
        """
        Initialize the WorkloadMetric.

        Args:
            requires_hooks (bool, default=False): Flag indicating if the metric requires hooks

        """
        self._requires_hooks = requires_hooks

    @abstractmethod
    def __call__(
        self, model: NeuroBenchModel, preds: Tensor, data: tuple[Tensor, Tensor]
    ) -> float:
        """
        Compute the workload metric.

        This method must be implemented by any subclass to define how the metric
        should be computed based on the model, predictions, and data.

        Args:
            model (NeuroBenchModel): The model whose performance is being evaluated.
            preds (Tensor): A tensor of model predictions.
            data (tuple[Tensor, Tensor]): A tuple containing the input data (Tensor)
            and the true labels (Tensor).

        Returns:
            float: The computed value of the workload metric.

        """
        pass

    @property
    def requires_hooks(self) -> bool:
        """
        Property indicating whether the metric requires hooks.

        Returns:
            bool: True if the metric requires hooks, False otherwise.

        """
        return self._requires_hooks


class AccumulatedMetric(WorkloadMetric):
    """
    Abstract base class for accumulated workload metrics.

    An accumulated metric tracks values over multiple batches or iterations and computes
    the final metric value after accumulating data. It extends the WorkloadMetric class
    and adds functionality for resetting and computing the accumulated metric over time.

    """

    def __init__(self, requires_hooks: bool = False):
        """
        Initialize the AccumulatedMetric.

        Args:
            requires_hooks (bool, default=False): Flag indicating if the metric requires hooks

        """
        super().__init__(requires_hooks)

    @abstractmethod
    def compute(self) -> float:
        """
        Compute the accumulated metric.

        This method must be implemented by any subclass to compute the accumulated
        value of the metric, typically after processing multiple batches.

        Returns:
            float: The computed accumulated metric value.

        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the accumulated state.

        This method must be implemented by any subclass to reset the metric's
        accumulated state.

        """
        pass
