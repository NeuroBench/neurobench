from neurobench.benchmarks.metrics.base.workload_metric import WorkloadMetric
from neurobench.benchmarks.metrics import workload as workload_metrics
from neurobench.benchmarks.workload_metrics import detect_activations_connections

requires_hooks = [
    "activation_sparsity",
    "number_neuron_updates",
    "synaptic_operations",
    "membrane_updates",
]


class WorkloadMetricManager:
    """Orchestrator for managing and executing workload metrics on a model."""

    def __init__(self, metric_list):
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
                metric_class = getattr(workload_metrics, item, None)
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
                self.metrics[item.__name__] = item()

        # Track whether hooks are required
        self.requires_hooks = any(
            metric_name in requires_hooks for metric_name in self.metrics.keys()
        )

    def register_hooks(self, model):
        """
        Register hooks on the model for metrics that require them.

        Args:
            model: The model on which the hooks will be registered.

        """
        if self.requires_hooks:
            detect_activations_connections(model)

    def cleanup_hooks(self, model):
        """
        Cleanup hooks on the model after metrics have been run.

        Args:
            model: The model on which the hooks will be cleaned up.

        """
        if self.requires_hooks:
            model.reset_hooks()

    def run_metrics(self, model, preds, data):
        """
        Executes all workload metrics on the provided model and data.

        Args:
            model: The model on which the metrics will be run.
            data: A tuple of inputs and targets.

        Returns:
            dict: A dictionary where the keys are metric names and the values are the results of the metric calculations.

        """

        results = {}

        for name, metric in self.metrics.items():
            try:
                results[name] = metric(model, preds, data)
            except Exception as e:
                print(f"Error running workload metric '{name}': {e}")
                results[name] = None  # Graceful fallback in case of errors

        return results
