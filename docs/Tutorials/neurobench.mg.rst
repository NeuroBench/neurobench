.. _mackey-glass-benchmark-tutorial:

=======================================
**Mackey Glass Benchmark Tutorial**
=======================================

This tutorial aims to provide insights into how the NeuroBench framework is organized and how you can utilize it to benchmark your models!

This is a static non-editable version. The editable version can be found at this link:
 * `Local Notebook <path/to/ipynb_file>`__
 
.. _about-mackey-glass:

**About Mackey Glass**
------------------------

The Mackey Glass task is a chaotic function prediction task. Unlike other tasks in NeuroBench, the Mackey Glass dataset is synthetic. Real-world data can be high-dimensional, requiring large networks for high accuracy, posing challenges for solution types with limited I/O support and network capacity, such as mixed-signal prototype solutions.

.. _dataset:

**Dataset**
------------

The Mackey Glass dataset is a one-dimensional nonlinear time-delay differential equation, where the evolution of the signal can be altered by various parameters. These parameters are defined in NeuroBench.

.. math:: \frac{dx}{dt} = \frac{\beta x(t-\tau)}{1 + x(t-\tau)^n} - \gamma x(t)

.. _benchmark-task:

**Benchmark Task**
-------------------

The task is a sequence-to-sequence prediction problem, similar to the primate reaching task in NeuroBench. The input sequence x is used to predict future values of the same sequence, y(t) = x(t). The input data is passed at a time step of :math:`\Delta t`, and the system's performance is tested in a multi-horizon prediction setting, predicting future values of the sequence at a rate of :math:`\Delta t`. The task's difficulty varies by adjusting the ratio between the integration time step :math:`\Delta t` and the timescale :math:`\tau` of the underlying dynamics.

.. container:: cell code

   .. code:: python

      import torch
      from torch.utils.data import Subset, DataLoader
      import pandas as pd
      from neurobench.datasets import MackeyGlass
      from neurobench.models import TorchModel
      from neurobench.benchmarks import Benchmark

.. _model-imports:

**Model Imports**
------------------

For this tutorial, we will use the example architecture included in the NeuroBench framework.

.. container:: cell code

   .. code:: python

      # This is the network we will use in this tutorial
      from neurobench.examples.mackey_glass.echo_state_network import EchoStateNetwork

.. _mackey-glass-parameters:

**Mackey Glass Parameters**
---------------------------

The Mackey Glass task is a synthetic dataset generated when calling the MackeyGlass function. Users must pass the parameters of the Mackey Glass function to define the output sequence. NeuroBench provides parameters that can be used to obtain the data.

.. container:: cell code

   .. code:: python

      # Mackey Glass parameters
      mg_parameters_file = "neurobench/datasets/mackey_glass_parameters.csv"
      mg_parameters = pd.read_csv(mg_parameters_file)

.. container:: cell code

   .. code:: python

      # Load the hyperparameters of the echo state networks found via random search
      esn_parameters = pd.read_csv("echo_state_network_hyperparameters.csv")

.. container:: cell code

   .. code:: python

      preprocessors = []
      postprocessors = []

.. container:: cell code

   .. code:: python

      # Benchmark run over 14 different series
      sMAPE_scores = []
      # Number of simulations to run for each time series
      repeat = 10

.. container:: cell code

   .. code:: python

      # Shift time series by 0.5 of its Lyapunov times for each independent run
      start_offset_range = torch.arange(0., 0.5 * repeat, 0.5)

.. container:: cell code

   .. code:: python

      for repeat_id in range(repeat):
          for series_id in range(len(mg_parameters)):
              tau = mg_parameters.tau[series_id]
              # Load data using the parameters loaded from the CSV file
              mg = MackeyGlass(tau=tau, lyaptime=mg_parameters.lyapunov_time[series_id], constant_past=mg_parameters.initial_condition[series_id], start_offset=start_offset_range[repeat_id].item(), bin_window=1)
              # Split test and train set
              train_set = Subset(mg, mg.ind_train)
              test_set = Subset(mg, mg.ind_test)
              # Index of the hyperparameters for the current time series
              ind_tau = esn_parameters.index[esn_parameters['tau'] == tau].tolist()[0]

              # Fitting Model
              seed_id = repeat_id
              # Load the model with the parameters loaded from esn_parameters
              esn = EchoStateNetwork(in_channels=1, reservoir_size=esn_parameters['reservoir_size'][ind_tau], input_scale=torch.tensor([esn_parameters['scale_bias'][ind_tau], esn_parameters['scale_input'][ind_tau]], dtype=torch.float64), connect_prob=esn_parameters['connect_prob'][ind_tau], spectral_radius=esn_parameters['spectral_radius'][ind_tau], leakage=esn_parameters['leakage'][ind_tau], ridge_param=esn_parameters['ridge_param'][ind_tau], seed_id=seed_id)

              esn.train()
              train_data, train_labels = train_set[:]
              warmup = 0.6  # in Lyapunov times
              warmup_pts = round(warmup * mg.pts_per_lyaptime)
              train_labels = train_labels[warmup_pts:]
              esn.fit(train_data, train_labels, warmup_pts)
              # Save the model for later use
              torch.save(esn, 'neurobench/examples/mackey_glass/model_data/esn.pth')

              # Load Model
              net = torch.load('neurobench/examples/mackey_glass/model_data/esn.pth')
              test_set_loader = DataLoader(test_set, batch_size=mg.testtime_pts, shuffle=False)
              # Wrap the model
              model = TorchModel(net)
              static_metrics = ["model_size", "connection_sparsity"]
              data_metrics = ["sMAPE", "activation_sparsity"]
              benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, data_metrics])
              results = benchmark.run()
              print(results)
              sMAPE_scores.append(results["sMAPE"])

      print("Average sMAPE score across all repeats and time series: ", sum(sMAPE_scores) / len(sMAPE_scores))
