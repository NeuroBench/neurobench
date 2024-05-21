================
Tutorial
================

This tutorial walks through the NeuroBench harness components and run setup using the Google Speech Commands (GSC) keyword classification task.
Each of the other benchmarks also has a notebook walkthrough, which can be found in the `NeuroBench examples directory <https://github.com/NeuroBench/neurobench/tree/main/neurobench/examples>`_.
The GSC task is presented here for its simplicity.

**Benchmark Task:**
The task is to classify keywords from the GSC dataset test split, after
training using the train and val splits.

First we will import the relevant libraries. These include the dataset,
pre- and post-processors, model wrapper, and benchmark object.

.. code:: python

    import torch
    # import the dataloader
    from torch.utils.data import DataLoader
    
    # import the dataset, preprocessors and postprocessors you want to use
    from neurobench.datasets import SpeechCommands
    from neurobench.preprocessing import S2SPreProcessor
    from neurobench.postprocessing import choose_max_count
    
    # import the NeuroBench wrapper to wrap the snnTorch model
    from neurobench.models import SNNTorchModel
    # import the benchmark class
    from neurobench.benchmarks import Benchmark

For this tutorial, we will make use of a simple feedforward SNN, written
using snnTorch.

.. code:: python

    from torch import nn
    import snntorch as snn
    from snntorch import surrogate
    
    beta = 0.9
    spike_grad = surrogate.fast_sigmoid()
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(20, 256),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(256, 256),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(256, 256),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(256, 35),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
    )

To get started, we will load our desired dataset in a dataloader. Note that any
torch.Dataloader can be used for the benchmark, it is not constrained to the datasets
available in the harness. Check out the `Tonic library <https://tonic.readthedocs.io/en/latest/#>`_
for an excellent resource for neuromorphic datasets!

.. code:: python

    test_set = SpeechCommands(path="data/speech_commands/", subset="testing")
    
    test_set_loader = DataLoader(test_set, batch_size=500, shuffle=True)

Here, we are loading a pre-trained model. The model is wrapped in the
SNNTorchModel wrapper, which includes boilerplate inference code and
interfaces with the top-level Benchmark class.

.. code:: python

    net.load_state_dict(torch.load("neurobench/examples/gsc/model_data/s2s_gsc_snntorch"))
    
    # Wrap our net in the SNNTorchModel wrapper
    model = SNNTorchModel(net)

Specify any pre-processors and post-processors you want to use. These
will be applied to your data before feeding into the model, and to the
output spikes respectively. Here, we are using the Speech2Spikes
pre-processor to convert the keyword audio data to spikes, and the
choose_max_count post-processor which returns a classification based on
the neuron with the greatest number of spikes.

.. code:: python

    preprocessors = [S2SPreProcessor()]
    postprocessors = [choose_max_count]

Next specify the metrics which you want to calculate. The metrics
include static metrics, which are computed before any model inference,
and workload metrics, which show inference results.

-  Footprint: Bytes used to store the model parameters and buffers.
-  Connection sparsity: Proportion of zero weights in the model.
-  Classification accuracy: Accuracy of keyword predictions.
-  Activation sparsity: Proportion of zero activations, averaged over
   all neurons, timesteps, and samples.
-  Synaptic operations: Number of weight-activation operations, averaged
   over keyword samples.

   -  Effective MACs: Number of non-zero multiply-accumulate synops,
      where the activations are not spikes with values -1 or 1.
   -  Effective ACs: Number of non-zero accumulate synops, where the
      activations are -1 or 1 only.
   -  Dense: Total zero and non-zero synops.

.. code:: python

    static_metrics = ["footprint", "connection_sparsity"]
    workload_metrics = ["classification_accuracy", "activation_sparsity", "synaptic_operations"]

Next, we instantiate the benchmark. We pass the model, the dataloader,
the preprocessors, the postprocessor and the list of the static and data
metrics which we want to measure:

.. code:: python

    benchmark = Benchmark(model, test_set_loader, 
                          preprocessors, postprocessors, [static_metrics, workload_metrics])

Now, let’s run the benchmark and print our results!

.. code:: python

    results = benchmark.run()
    print(results)

Expected output: {‘footprint’: 583900, ‘connection_sparsity’: 0.0,
‘classification_accuracy’: 0.8484325295196562, ‘activation_sparsity’:
0.9675956131759854, ‘synaptic_operations’: {‘Effective_MACs’: 0.0,
‘Effective_ACs’: 3556689.9895502045, ‘Dense’: 29336955.0}}