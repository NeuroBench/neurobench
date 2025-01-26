from neurobench.metrics import static as static_metrics
from neurobench.metrics.abstract import StaticMetric
from neurobench.metrics.utils.utils import convert_to_class_name
from neurobench.models import NeuroBenchModel
from typing import List, Dict


class StaticMetricManager:
    """Orchestrator for managing and executing static metrics on a model."""

    def __init__(self, metric_list: List):
        """
        Initializes the orchestrator with a list of static metrics.

        Args:
            metric_list (List): A list of metric identifiers, either as strings corresponding to internal
                                metric names or as custom metric objects.

        Raises:
            TypeError: If a metric object does not inherit from StaticMetric.
            ValueError: If a string-based metric is not found in the internal static_metrics module.

        """
        self.metrics = {}

        for item in metric_list:
            if isinstance(item, str):
                class_name = convert_to_class_name(item)
                metric_class = getattr(static_metrics, class_name, None)
                print(
                    f"[DEPRECATION WARNING]: Using string-based metric names ('{item}') is deprecated "
                    "and will be removed in a future release. "
                    "Please update your code to use the corresponding metric class directly instead. "
                    f"For example, replace the string name with the class '{class_name}'."
                )
                if metric_class is None:
                    raise ValueError(
                        f"Metric class '{class_name}' not found in the 'static_metrics' module."
                    )
                if not issubclass(metric_class, StaticMetric):
                    raise TypeError(
                        f"The metric class '{class_name}' must inherit from StaticMetric."
                    )
                self.metrics[item] = metric_class()  # Instantiate the metric class
            else:
                # Validate custom metric object
                if not issubclass(item, StaticMetric):
                    raise TypeError(
                        f"Custom metric '{item.__name__}' must inherit from StaticMetric."
                    )
                self.metrics[item.__name__] = item()

    def run_metrics(self, model: NeuroBenchModel) -> Dict:
        """
        Executes all static metrics on the provided model.

        Args:
            model: The model on which the metrics will be run.

        Returns:
            Dict: A dictionary where the keys are metric names and the values are the results of the metric calculations.

        Raises:
            Exception: If a metric computation fails, it is caught and logged with an error message.

        """
        results = {}
        for name, metric in self.metrics.items():
            try:
                results[name] = metric(model)
            except Exception as e:
                print(f"Error running metric '{name}': {e}")
                results[name] = None  # Graceful fallback in case of errors
        return results
