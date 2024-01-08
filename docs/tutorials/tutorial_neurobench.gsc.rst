=============================================
**Google Speech Commands Benchmark Tutorial**
=============================================

This tutorial aims to provide an insight on how the NeuroBench framework is organized and how you can use it to benchmark your own models!

This is a static non-editable version. The editable version can be found at this link:
 * `Local Notebook <path/to/ipynb_file>`__
 
**About Google Speech Commands**
----------------------------------

Google Speech Commands is a keyword spotting task. Voice commands represent a natural and easily accessible modality for human-machine interaction. Keyword detection, in particular, is frequently employed in edge devices that operate in always-listening, wake-up situations, where it triggers more computationally demanding processes such as automatic speech recognition. Keyword spotting finds application in activating voice assistants, speech data mining, audio indexing, and phone call routing. Given that it generally operates in always-on and battery-powered edge scenarios, keyword detection represents a pertinent benchmark for energy-efficient neuromorphic solutions.

**Dataset**
------------

The Google Speech Commands dataset (V2) is a commonly used dataset in assessing the performance of keyword spotting algorithms. The dataset consists of 105,829 1 second utterances of 35 different words from 2,618 distinct speakers. The data is encoded as linear 16-bit, single-channel, pulse code modulated values, at a 16 kHz sampling frequency.

**Benchmark Task**
-------------------

The goal is to develop a model that trains using the designated train and validation sets, followed by an evaluation of generalization to a separate test set. The task is a classification task.

First we will import the relevant libraries. These include the datasets, preprocessors and accumulators. To ensure your model to be compatible with the NeuroBench framework, we will import the wrapper for snnTorch models. This wrapper will not change your model. Finally, we import the Benchmark class, which will run the benchmark and calculate your metrics.

.. code:: python

   import torch
   # import the dataloader
   from torch.utils.data import DataLoader

   # import the dataset, preprocessors and accumulators you want to use
   from neurobench.datasets import SpeechCommands
   from neurobench.preprocessing import S2SProcessor
   from neurobench.accumulators import choose_max_count

   # import the NeuroBench wrapper to wrap your snnTorch model for usage in the NeuroBench framework
   from neurobench.models import SNNTorchModel
   # import the benchmark class
   from neurobench.benchmarks import Benchmark

For this tutorial, we will make use of the example architecture that is included in the NeuroBench framework.

.. code:: python

   # this is the network we will be using in this tutorial
   from neurobench.examples.gsc.SNN import net

To get started, we will load our desired dataset in a dataloader:

.. code:: python

   test_set = SpeechCommands(path="data/speech_commands/", subset="testing")

   test_set_loader = DataLoader(test_set, batch_size=500, shuffle=True)

Next, load our model and wrap it in the corresponding NeuroBench wrapper. At the time of writing this tutorial, (V1.0) snnTorch is the only supported framework, therefore, we will wrap our snnTorch model in the SNNTorchModel() wrapper. Note that any framework based on PyTorch can be wrapped in a TorchModel() wrapper, and the neuron layers can be added with add_activation_module().

.. code:: python

   net.load_state_dict(torch.load("neurobench/examples/gsc/model_data/s2s_gsc_snntorch", map_location=torch.device('cpu')))
   # Wrap our net in the SNNTorchModel wrapper
   model = SNNTorchModel(net)

Specify the preprocessor and postprocessor want to use. These will be applied to your data before feeding into the model, and to the output spikes respectively. Available preprocessors and postprocessors can be found in neurobench/preprocessors and neurobench/accumulators respectively.

.. code:: python

   preprocessors = [S2SProcessor()]
   postprocessors = [choose_max_count]

Next specify the metrics which you want to calculate. The available metrics (V1.0 release) are:

**Static Metrics:**

- footprint
- connection_sparsity
- parameter_count
- Model Excecution Rate

**Data Metrics:**

- activation_sparsity
- synaptic_operations
- classification_accuracy
- coco_map
- mse
- r2
- smape

More accuracy metrics are available, for which the user is recommended to consult the documentation. Note that the Model Excecution Rate is not returned by the famework, but reported by the user. Execution rate, in Hz, of the model computation based on forward inference passes per second, measured in the time-stepped simulation timescale. More explanation on the metrics can be found on `neurobench.ai <https://neurobench.ai/>`. 

.. code:: python

   static_metrics = ["footprint"]
   data_metrics = ["classification_accuracy"]

Next, we instantiate the benchmark. We have to specify the model, the dataloader, the preprocessors, the postprocessor and the list of the static and data metrics which we want to measure:

.. code:: python

   benchmark = Benchmark(model, test_set_loader, preprocessors, postprocessors, [static_metrics, data_metrics])

Now, let's run the benchmark and print our results!

.. code:: python

   results = benchmark.run()
   print(results)

