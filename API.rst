NeuroBench API: Developent Version
==================================

This is a brief overview of the state of the NeuroBench API, the
specifications NeuroBench components are expected to conform to, and any
exceptions or errata that have come up during development.

Example Python snippets may be shown here and will be kept up-to-date.

Overview
--------

+----------------+
| **Components** |
+================+
| Data           |
+----------------+
| Dataset        |
+----------------+
| Processor      |
+----------------+
| Accumulator    |
+----------------+
| Model          |
+----------------+
| Metrics        |
+----------------+
| Benchmark      |
+----------------+

.. raw:: html

   <div class="page"/>

Specifications
--------------

**Data:**
~~~~~~~~~

::

   Format:
       tensor: A PyTorch tensor of shape (batch, timesteps, features*), where features* can be any number of dimensions.

**Dataset:**
~~~~~~~~~~~~

::

   Output:
       (data, targets): A tuple of PyTorch tensors. The first dimension (batch) is expected to match.
    
   Alternatively, if other metadata is required from the dataset, like in 1MP object detection, then there can be a 3-tuple with kwargs.

   Output:
       (data, targets, kwargs): kwargs is a dictionary of metadata.

**Processor:**
~~~~~~~~~~~~~~

Processing data / preprocessing.

::

   Input:
       (data, targets): A tuple of PyTorch tensors. The first dimension (batch) is expected to match.
   Output:
       (data, targets): A tuple of PyTorch tensors. The first dimension (batch) is expected to match.

.. code:: python

   class Processor(NeuroBenchProcessor):
       def __init__(self):
           ...
       def __call__(self, dataset):
           ...

   alg = Processor()
   new_dataset = alg(dataset) # dataset: (data, targets)

**Accumulator:**
~~~~~~~~~~~~~~~~

Accumulating predictions / postprocessing.

::

   Input:
       preds: A PyTorch tensor.
   Output:
       results: A PyTorch tensor. Accumulators may be chained together. Final shape is expected to match the data targets for comparison.

.. code:: python

    class Accumulator(NeuroBenchAccumulator):
        def __init__(self):
            ...
        def __call__(self, preds):
            ...

   alg = Accumulator()
   model = NeuroBenchModel(...)
   preds = model(data) # data: (batch, timesteps, features*)
   results = alg(preds)

**Model:**
~~~~~~~~~~

::

   Input:
       data: A PyTorch tensor of shape (batch, timesteps, features*)
   Output:
       preds: A PyTorch tensor. Can either be the final shape to be compared with targets or an arbitrary shape to be postprocessed by Accumulator(s).

.. code:: python

    class SNNTorchModel(NeuroBenchModel):
        def __init__(self, net):
            ...
        def __call__(self, batch):
            ...

   model = SNNTorchModel(net)
   preds = model(batch)

**Metrics:**
~~~~~~~~~~~~

There are two types of metrics: *static* and *data*. Static metrics can
be computed using the model alone, while data metrics require the model
predictions and the targets as well.

Static metrics are stateless functions.

Data metrics can also be stateless functions (in which case they are accumulated over batched evaluation via mean),
or they can be stateful subclasses of AccumulatedMetric.

::

   **Static Metrics:**
   Input:
       model: A NeuroBenchModel object.
   Output:
       result: Any type. The result of the metric.

::

   **Data Metrics:**
   Input:
       model: A NeuroBenchModel object.
       preds: A PyTorch tensor. To be compared with targets.
       data: Tuple of (data, targets). 
   Output:
       result: A float or int.

.. code:: python

    def static_metric(model):
        ...

    def data_metric(model, preds, data):
        # must return an int or float to be accumulated with mean
        return compare(preds, data[1])

    class data_metric_with_state(AccumulatedMetric):
        def __init__(self):
            ...
    
    def __call__(self, model, preds, data):
        # accumulate state from this batch
        return self.compute()
    
    def compute():
        # compute metric from accumulated state

**Benchmark:**
~~~~~~~~~~~~~~

::

   Input:
       model: The NeuroBenchModel to be tested.
       dataloader: A PyTorch DataLoader which loads the evaluation dataset.
       processors: A list of Processors.
       accumulators: A list of Accumulators.
       metric_list: [[static_metrics], [data_metrics]], where each are strings. The names of the metric will be used to call it from the metrics file. User defined metrics should be discouraged.
   Output:
       results: A dict of {metric: result}.

.. code:: python

   model = TorchModel(net)
   test_set = NeuroBenchDataset(...)
   test_set_loader = DataLoader(test_set, batch_size=16, shuffle=False)
   processors = [Processor1(), Processor2()]
   accumulators = [Accumulator1()]
   static_metrics = ["model_size", "connection_sparsity"]
   data_metrics = ["accuracy", "activation_sparsity"]

   benchmark = Benchmark(
       model, 
       test_set_loader,
       processors,
       accumulators, 
       [static_metrics, data_metrics]
   )
   results = benchmark.run()

Known Errata
------------

Any anomalies that break the high-level API will be noted here but
attempts will be made to keep this to a minimum.
