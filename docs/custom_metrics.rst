Custom Metrics
======================

This guide explains how to create custom metrics using the NeuroBench framework. Metrics are categorized into two types:
**Static Metrics** and **Workload Metrics**. Each type has a specific purpose and use case, and
this document provides examples for defining metrics for each type.

Metric Types
------------

The following metric types are available:

- **Static Metrics**: Evaluate fixed properties of the model, such as the number of parameters.
- **Workload Metrics**: Evaluate the model's performance during inference execution.

By default, workload metrics will averaged across batched inference. Metrics like classification accuracy can be joined this way.

Other workload metrics may depend on all data and inferences and cannot be averaged across batches, such as an R^2 score. These workload metrics should be *AccumulatedMetrics*, which is a subclass of workload metrics which stores performance over multiple batches and computes final metric values once data is processed.


Static Metric Abstract Base Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: neurobench.metrics.abstract.static_metric
    :members:
    :undoc-members:
    :show-inheritance:

Workload Metric Abstract Base Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: neurobench.metrics.abstract.workload_metric
    :members: WorkloadMetric
    :undoc-members:
    :exclude-members: requires_hooks

Accumulated Metric Abstract Base Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: neurobench.metrics.abstract.workload_metric
    :members: AccumulatedMetric
    :undoc-members:
    :exclude-members: requires_hooks

Defining Custom Metrics
-----------------------

Here’s how you can define your own metrics for each type:

Static Metrics
^^^^^^^^^^^^^^

Static metrics are used to evaluate fixed properties of a model, such as the total number of parameters.

Example:
    .. code-block:: python

        from neurobench.metrics.abstract import StaticMetric

        class ParameterCountMetric(StaticMetric):
            """
            Metric to count the total number of parameters in a model.
            """

            def __call__(self, model):
                return sum(p.numel() for p in model.parameters())

Workload Metrics
^^^^^^^^^^^^^^^^

Workload metrics (which are not Accumulated) evaluate the model’s performance for each batch of input data, and the result is averaged across batches.

Example:
    .. code-block:: python

        from neurobench.metrics.abstract import WorkloadMetric

        class AccuracyMetric(WorkloadMetric):
            """
            Metric to compute accuracy for a single batch.
            """

            def __call__(self, model, preds, data):
                inputs, labels = data
                correct = (preds.argmax(dim=1) == labels).sum().item()
                return correct / len(labels) * 100

Accumulated Metrics
^^^^^^^^^^^^^^^^^^^

Accumulated metrics are a type of workload metric which stores performance information over multiple batches and compute a final result. These should be used when the metric should not be averaged across batches.

Example:
    .. code-block:: python

        from neurobench.metrics.abstract import AccumulatedMetric

        class MaximumLossMetric(AccumulatedMetric):
            """
            Metric to compute the maximum loss over multiple batches.
            """

            def __init__(self):
                super().__init__()
                self.max_loss = 0.0
                self.num_batches = 0

            def __call__(self, model, preds, data):
                _, labels = data
                loss = loss_function(preds, labels).item()
                if loss > self.max_loss:
                    self.max_loss = loss
                self.num_batches += 1

            def compute(self):
                return self.max_loss

            def reset(self):
                self.max_loss = 0.0
                self.num_batches = 0

Plugging Metrics into the Benchmark
^^^^^^^^^^^^^^^^^^^

The `Benchmark` class expects metrics to be passed as lists grouped by type.
You also need to configure any `postprocessors` if required by your model.

Here’s an example of integrating both built-in and custom metrics into the `Benchmark` class:

Example:
    .. code-block:: python

        from neurobench.metrics import Benchmark
        from neurobench.postprocessors import ChooseMaxCount

        # Define datasets and the model
        model = Model()  # Replace with your model instance
        test_set_loader = DataLoader()  # Replace with your DataLoader instance

        # Define postprocessors (if applicable)
        postprocessors = [ChooseMaxCount()]

        # Define metrics
        static_metrics = [ParameterCountMetric] # Replace with your custom static metric. Do not initialize the classes.
        workload_metrics = [AccuracyMetric, MaximumLossMetric]  # Replace with your custom workload and accumulated metrics. Do not initialize the classes.

        # Create the Benchmark instance
        benchmark = Benchmark(
            model=model,
            dataloader=test_set_loader,
            postprocessors=postprocessors,
            metrics=[static_metrics, workload_metrics]
        )

        # Run the benchmark
        results = benchmark.run(verbose=True)

        # Access results
        print(results)




Summary of Custom Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^

The following table summarizes the available metric types:

.. list-table:: Metric Types
   :header-rows: 1

   * - Metric Type
     - Base Class
     - Key Methods
     - Use Case
   * - Static Metric
     - ``StaticMetric``
     - ``__call__``
     - Evaluate static properties of the model.
   * - Workload Metric
     - ``WorkloadMetric``
     - ``__call__``
     - Averages over batched inference.
   * - Accumulated Metric
     - ``AccumulatedMetric``
     - ``__call__``, ``compute``, ``reset``
     - Accumulates performance info over multiple batches.