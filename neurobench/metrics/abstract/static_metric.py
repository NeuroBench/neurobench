from abc import ABC, abstractmethod
from neurobench.models import NeuroBenchModel


class StaticMetric(ABC):
    """
    Abstract base class for static metrics.

    A static metric computes a value based on the model, and is designed to operate
    without the need for batch processing or iterative accumulation over multiple
    samples. This metric is expected to return a single value when called, which is
    typically computed over the entire model.

    """

    @abstractmethod
    def __call__(self, model: NeuroBenchModel) -> float:
        """
        Compute the static metric for the given model.

        This abstract method must be implemented by any subclass to define
        how the metric should be computed based on the model.

        Args:
            model (NeuroBenchModel): The model whose performance or properties the metric evaluates.

        Returns:
            float: The computed value of the metric for the model.

        Notes:
            - The method should return a single float value representing the
              result of the metric computation. It does not require batch
              processing or tracking multiple samples.
            - This is intended for metrics that compute a single, static
              evaluation of a model, such as those based on its architecture,
              weights, or overall performance.

        """
        pass
