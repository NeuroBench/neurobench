.. _primate-reaching-benchmark-tutorial:

============================================
**Primate Reaching Benchmark Tutorial**
============================================

This tutorial aims to provide insights into how the NeuroBench framework is organized and how you can utilize it to benchmark your models!

This is a static non-editable version. The editable version can be found at this link:
 * `Local Notebook <path/to/ipynb_file>`__
 
.. _about-primate-reaching:

**About Primate Reaching**
---------------------------

There is significant interest in models that not only take inspiration from but also strive to accurately replicate features of biological computation. The study of these models presents opportunities to gain a more comprehensive understanding of sensorimotor behavior and the underlying computational primitives that facilitate them, which can be used to develop closed-loop and model-predictive control tasks essential for controlling future robotic agents. Additionally, this research has implications for the development of wearable or implantable neuro-prosthetic devices that can accurately predict motor activity from neural or muscle signals. Hence, motor prediction is important.

.. _dataset:

**Dataset**
------------

The dataset that we utilize in this study consists of multi-channel recordings obtained from the sensorimotor cortex of two non-human primates (NHP) during self-paced reaching movements towards a grid of targets. The variable x is represented by threshold crossing times (or spike times) and sorted units for each of the recording channels. The target y is represented by 2-dimensional position coordinates of the fingertip of the reaching hand, sampled at a frequency of 250 Hz. The complete dataset contains 37 sessions spanning 10 months for NHP-1 and 10 sessions from NHP-2 spanning one month. For this study, three sessions from each NHP were selected to include the entire recording duration, resulting in a total of 6774 seconds of data.

.. _benchmark-task:

**Benchmark Task**
-------------------

In the context of predictive modeling, time series prediction is a task that entails the forecasting of one or more observations of a target variable, y, at some point between the current time, :math:`t`, and a future time, :math:`t` + :math:`t_f`, by utilizing a sequence of another variable, :math:`x`, from the past, {:math:`x(t − th), . . . , x(t)`}. In the primate reaching task, the goal is to predict the :math:`X` and :math:`Y` components of finger velocity, :math:`y`, from past neural data, :math:`x`, with a minimum frequency of 10 Hz. The model architecture may be trained separately for each session to account for inter-day neural variability. The training data is divided into either 50% or 80% for training, while the remaining split is distributed equally between validation and testing. This allows for testing of the model’s generalization capabilities with varying data sizes and comparison with related work in the field.

.. container:: cell code

   .. code:: python

      from neurobench.datasets import PrimateReaching
      from neurobench.models.torch_model import TorchModel
      from neurobench.benchmarks import Benchmark

.. container:: cell code

   .. code:: python

      from neurobench.examples.primate_reaching.ANN import ANNModel

.. container:: cell code

   .. code:: python

      # The dataloader and preprocessor have been combined together into a single class
      primate_reaching_dataset = PrimateReaching(file_path="data/primate_reaching/PrimateReachingDataset/", filename="indy_20170131_02.mat",
                                                 num_steps=7, train_ratio=0.8, mode="3D", model_type="ANN")
      test_set = primate_reaching_dataset.create_dataloader(primate_reaching_dataset.ind_test, batch_size=256, shuffle=True)

.. container:: cell code

   .. code:: python

      net = ANNModel(input_dim=primate_reaching_dataset.input_feature_size * primate_reaching_dataset.num_steps,
                     layer1=32, layer2=48, output_dim=2, dropout_rate=0.5)

      # Give the user the option to load their pretrained weights
      # net.load_state_dict(torch.load("neurobench/examples/primate_reaching/model_data/model_parameters.pth"))

      model = TorchModel(net)

.. container:: cell code

   .. code:: python

      preprocessors = []
      postprocessors = []

.. container:: cell code

   .. code:: python

      static_metrics = ["model_size"]
      data_metrics = ["r2", "activation_sparsity"]

.. container:: cell code

   .. code:: python

      # Benchmark expects the following:
      benchmark = Benchmark(model, test_set, [], [], [static_metrics, data_metrics])
      results = benchmark.run()
      print(results)
