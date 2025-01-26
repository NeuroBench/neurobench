from collections import defaultdict

from neurobench.metrics.abstract.workload_metric import (
    WorkloadMetric,
    AccumulatedMetric,
)
from neurobench.metrics import workload as workload_metrics
from neurobench.metrics.utils.utils import convert_to_class_name

from neurobench.models import NeuroBenchModel
from torch import Tensor
from typing import List, Dict


class WorkloadMetricManager:
    """Orchestrator for managing and executing workload metrics on a model."""

    def __init__(self, metric_list: List):
        """
        Initializes the orchestrator with a list of workload metrics.

        Args:
            metric_list (list): A list of metric identifiers, either as strings corresponding to internal
                                metric names or as custom metric objects.

        Raises:
            ValueError: If a string-based metric is not found in the internal workload_metrics module.
            TypeError: If a custom metric object does not inherit from WorkloadMetric.

        """
        self.metrics = {}

        # Store workload metrics
        for item in metric_list:
            if isinstance(item, str):
                class_name = convert_to_class_name(item)
                metric_class = getattr(workload_metrics, class_name, None)
                print(
                    f"[DEPRECATION WARNING]: Using string-based metric names ('{item}') is deprecated "
                    "and will be removed in a future release. "
                    "Please update your code to use the corresponding metric class directly instead. "
                    f"For example, replace the string name with the class '{class_name}'."
                )
                if metric_class is None:
                    raise ValueError(
                        f"Metric '{item}' not found in the 'workload_metrics' module."
                    )
                if not issubclass(metric_class, WorkloadMetric):
                    raise TypeError(
                        f"The metric class '{item}' must inherit from WorkloadMetric."
                    )
                self.metrics[item] = metric_class()
            else:
                # Validate custom metric object
                if not issubclass(item, WorkloadMetric):
                    raise TypeError(
                        f"Custom metric '{item.__name__}' must inherit from WorkloadMetric."
                    )
                self.metrics[item.__name__]: WorkloadMetric = item()

        # Track whether hooks are required
        self.requires_hooks = any(
            metric.requires_hooks for metric in self.metrics.values()
        )

        self.results = defaultdict(float)

    def clean_results(self) -> None:
        """Reset the results dictionary to zero."""
        self.results = defaultdict(float)

    def register_hooks(self, model: NeuroBenchModel) -> None:
        """
        Register hooks on the model for metrics that require them.

        Args:
            model: The model on which the hooks will be registered.

        """
        if self.requires_hooks:
            model.register_hooks()

    def reset_hooks(self, model: NeuroBenchModel) -> None:
        """
        Cleanup hooks on the model after metrics have been run.

        Args:
            model: The model on which the hooks will be cleaned up.

        """
        if self.requires_hooks:
            model.reset_hooks()

    def cleanup_hooks(self, model: NeuroBenchModel) -> None:
        """
        Cleanup hooks on the model after metrics have been run.

        Args:
            model: The model on which the hooks will be cleaned up.

        """
        if self.requires_hooks:
            model.cleanup_hooks()

    def close_hooks(self, model: NeuroBenchModel) -> None:
        """Close hooks on the model after metrics have been run."""
        if self.requires_hooks:
            model.close_hooks()

    def initialize_metrics(self) -> None:
        """Initialize the metrics for a new workload run."""

        for m in self.metrics.keys():
            if isinstance(self.metrics[m], AccumulatedMetric):
                self.metrics[m].reset()

    def run_metrics(
        self,
        model: NeuroBenchModel,
        preds: Tensor,
        data: tuple[Tensor, Tensor],
        batch_size: int,
        dataset_len: int,
    ) -> Dict:
        """
        Executes all workload metrics on the provided model and data.

        Args:
            model: The model on which the metrics will be run.
            preds: A tensor of model predictions.
            data: A tuple of inputs and targets.
            batch_size: The size of the batch.
            dataset_len: The length of the dataset.

        Returns:
            Dict: A dictionary where the keys are metric names and the values are the results of the metric calculations.

        """

        for name, metric in self.metrics.items():
            try:
                result = metric(model, preds, data)

                # Handle AccumulatedMetric separately
                if isinstance(metric, AccumulatedMetric):
                    self.results[name] = result
                else:
                    # Accumulate results, normalizing by dataset length
                    self.results[name] += (result * batch_size) / dataset_len

            except Exception as e:
                print(f"Error running workload metric '{name}': {e}")
                self.results[name] = 0.0  # Graceful fallback in case of errors

        return self.results
