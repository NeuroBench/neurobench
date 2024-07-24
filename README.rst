.. image:: https://github.com/NeuroBench/neurobench/blob/main/docs/_static/neurobench_banner_light.jpeg?raw=true
        :align: center
        :width: 800

============
Introduction
============


A harness for running evaluations on
`NeuroBench <https://neurobench.ai>`__ algorithm benchmarks.

NeuroBench is a community-driven project, and we welcome further
development from the community. If you are interested in developing
extensions to features, programming frameworks, or metrics and tasks,
please see the `Contributing Guidelines <https://neurobench.readthedocs.io/en/latest/contributing.html>`__.

NeuroBench Structure
---------------------

NeuroBench contains the following sections:

.. list-table:: 
   :widths: 20 60

   * - **Section**
     - **Description**
   * - `neurobench.benchmarks <https://neurobench.readthedocs.io/en/latest/neurobench.benchmarks.html>`__
     - Neurobench benchmarks, including data metrics and static metrics
   * - `neurobench.datasets <https://neurobench.readthedocs.io/en/latest/neurobench.datasets.html>`__
     - Neurobench benchmark datasets
   * - `neurobench.models <https://neurobench.readthedocs.io/en/latest/neurobench.models.html>`__
     - Neurobench framework for Torch and SNNTorch models
   * - `neurobench.preprocessing <https://neurobench.readthedocs.io/en/latest/neurobench.preprocessing.html>`__
     - Pre-processing of data, conversion to spikes
   * - `neurobench.postprocessing <https://neurobench.readthedocs.io/en/latest/neurobench.postprocessing.html>`__
     - Post-processors take the spiking output from the models and provide several methods of combining them

Installation
------------

Install from PyPI:

::

   pip install neurobench

Benchmarks
----------

The following benchmarks are currently available:

v1.0 benchmarks
~~~~~~~~~~~~~~~
- Keyword Few-shot Class-incremental Learning (FSCIL)
- Event Camera Object Detection
- Non-human Primate (NHP) Motor Prediction
- Chaotic Function Prediction

Additional benchmarks
~~~~~~~~~~~~~~~~~~~~~
- DVS Gesture Recognition
- Google Speech Commands (GSC) Classification
- Neuromorphic Human Activity Recognition (HAR)

Getting started
---------------

Example benchmark scripts can be found under the ``neurobench/examples`` folder. 
(`https://github.com/NeuroBench/neurobench/tree/main/neurobench/examples/ <https://github.com/NeuroBench/neurobench/tree/main/neurobench/examples/>`__)

In general, the design flow for using the framework is as follows:

1. Train a network using the train split from a particular dataset.
2. Wrap the network in a ``NeuroBenchModel``.
3. Pass the model, evaluation split dataloader, pre-/post-processors,
   and a list of metrics to the ``Benchmark`` and ``run()``.

Documentation for the framework interfaces can found in the `API Overview <https://neurobench.readthedocs.io/en/latest/api.html>`__.

Development
-----------

If you clone the repo directly for development, `poetry <https://pypi.org/project/poetry/>`__ 
can be used to maintain a virtualenv consistent with a deployment environment. In the
root directory run:

::

   pip install poetry
   poetry install

Poetry requires python >=3.9. Installation should not take more than a few minutes.

End-to-end examples can be run from the poetry environment. As a demo, try the 
Google Speech Commands keyword classification benchmark:

::

   # ANN Benchmark Example
   poetry run python neurobench/examples/gsc/benchmark_ann.py
   
   # Expected results:
   # {'footprint': 109228, 'connection_sparsity': 0.0,
   # 'classification_accuracy': 0.8653339397251905, 'activation_sparsity': 0.3854464619019532, 
   # 'synaptic_operations': {'Effective_MACs': 1749994.1556565198, 'Effective_ACs': 0.0, 'Dense': 1902179.0}}


   # SNN Benchmark Example
   poetry run python neurobench/examples/gsc/benchmark_snn.py
   
   # Expected results:
   # {'footprint': 583900, 'connection_sparsity': 0.0,
   # 'classification_accuracy': 0.8484325295196562, 'activation_sparsity': 0.9675956131759854, 
   # 'synaptic_operations': {'Effective_MACs': 0.0, 'Effective_ACs': 3556689.9895502045, 'Dense': 29336955.0}}

These demos should download the dataset, then run in a couple minutes. Other baseline result scripts and notebook
tutorials are available in the ``neurobench/examples`` folder.

Developers
----------

NeuroBench is a collaboration between industry and academic engineers
and researchers. This framework is currently maintained by `Jason
Yik <https://www.linkedin.com/in/jasonlyik/>`__, `Noah
Pacik-Nelson <https://www.linkedin.com/in/noah-pacik-nelson/>`__, and
`Korneel Van den
Berghe <https://www.linkedin.com/in/korneel-van-den-berghe/>`__, and
there have been technical contributions from many others. A
non-exhaustive list includes Gregor Lenz, Denis Kleyko, Younes
Bouhadjar, Paul Hueber, Vincent Sun, Biyan Zhou, George Vathakkattil
Joseph, Douwe den Blanken, Maxime Fabre, Shenqi Wang, Guangzhi Tang,
Anurag Kumar Mishra, Soikat Hasan Ahmed, Benedetto Leto, Aurora Micheli,
Tao Sun.

Contributing
------------

If you are interested in helping to build this framework, please see the
`Contribution Guidelines <https://neurobench.readthedocs.io/en/latest/contributing.html>`__.

Citation
--------

If you use this framework in your research, please cite the following
preprint article:

::

   @misc{yik2024neurobench,
      title={NeuroBench: A Framework for Benchmarking Neuromorphic Computing Algorithms and Systems}, 
      author={Jason Yik and Korneel Van den Berghe and Douwe den Blanken and Younes Bouhadjar and Maxime Fabre and Paul Hueber and Denis Kleyko and Noah Pacik-Nelson and Pao-Sheng Vincent Sun and Guangzhi Tang and Shenqi Wang and Biyan Zhou and Soikat Hasan Ahmed and George Vathakkattil Joseph and Benedetto Leto and Aurora Micheli and Anurag Kumar Mishra and Gregor Lenz and Tao Sun and Zergham Ahmed and Mahmoud Akl and Brian Anderson and Andreas G. Andreou and Chiara Bartolozzi and Arindam Basu and Petrut Bogdan and Sander Bohte and Sonia Buckley and Gert Cauwenberghs and Elisabetta Chicca and Federico Corradi and Guido de Croon and Andreea Danielescu and Anurag Daram and Mike Davies and Yigit Demirag and Jason Eshraghian and Tobias Fischer and Jeremy Forest and Vittorio Fra and Steve Furber and P. Michael Furlong and William Gilpin and Aditya Gilra and Hector A. Gonzalez and Giacomo Indiveri and Siddharth Joshi and Vedant Karia and Lyes Khacef and James C. Knight and Laura Kriener and Rajkumar Kubendran and Dhireesha Kudithipudi and Yao-Hong Liu and Shih-Chii Liu and Haoyuan Ma and Rajit Manohar and Josep Maria Margarit-Taulé and Christian Mayr and Konstantinos Michmizos and Dylan Muir and Emre Neftci and Thomas Nowotny and Fabrizio Ottati and Ayca Ozcelikkale and Priyadarshini Panda and Jongkil Park and Melika Payvand and Christian Pehle and Mihai A. Petrovici and Alessandro Pierro and Christoph Posch and Alpha Renner and Yulia Sandamirskaya and Clemens JS Schaefer and André van Schaik and Johannes Schemmel and Samuel Schmidgall and Catherine Schuman and Jae-sun Seo and Sadique Sheik and Sumit Bam Shrestha and Manolis Sifalakis and Amos Sironi and Matthew Stewart and Kenneth Stewart and Terrence C. Stewart and Philipp Stratmann and Jonathan Timcheck and Nergis Tömen and Gianvito Urgese and Marian Verhelst and Craig M. Vineyard and Bernhard Vogginger and Amirreza Yousefzadeh and Fatima Tuz Zohora and Charlotte Frenkel and Vijay Janapa Reddi},
      year={2024},
      eprint={2304.04640},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
   }
