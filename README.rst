============
Introduction
============

A harness for running evaluations on
`NeuroBench <https://neurobench.ai>`__ algorithm benchmarks.

This framework is in a beta state and is still under active development.
Currently, only pytorch-based models and frameworks are supported.
Extension of the harness to cover system track benchmarks in the future
is planned.

NeuroBench is a community-driven project, and we welcome further
development from the community. If you are interested in developing
extensions to features, programming frameworks, or metrics and tasks,
please see the `contributing guidelines <CONTRIBUTING.rst>`__.

NeuroBench Structure
---------------------

NeuroBench contains the following sections:

.. list-table:: 
   :widths: 20 60

   * - **Section**
     - **Description**
   * - `neurobench.benchmarks <docs/neurobench.benchmarks.rst>`__
     - Neurobench benchmarks, including data metrics and static metrics
   * - `neurobench.datasets <docs/neurobench.datasets.rst>`__
     - Neurobench benchmark datasets
   * - `neurobench.models <docs/neurobench.models.rst>`__
     - Neurobench framework for Torch and SNNTorch models
   * - `neurobench.preprocessing <docs/neurobench.preprocessing.rst>`__
     - Pre-processing of data, conversion to spikes
   * - `neurobench.postprocessing <docs/neurobench.postprocessing.rst>`__
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
- `Keyword Few-shot Class-incremental Learning (FSCIL) <neurobench/examples/mswc_fscil>`__
- `Event Camera Object Detection <neurobench/examples/obj_detection>`__
- `Non-human Primate (NHP) Motor Prediction <neurobench/examples/primate_reaching>`__
- `Chaotic Function Prediction <neurobench/examples/mackey_glass>`__

Additional benchmarks
~~~~~~~~~~~~~~~~~~~~~
- `DVS Gesture Recognition <neurobench/examples/dvs_gesture>`__
- `Google Speech Commands (GSC) Classification <neurobench/examples/gsc>`__
- `Neuromorphic Human Activity Recognition (HAR) <neurobench/examples/nehar>`__

Getting started
---------------

Example benchmark scripts can be found under the ``neurobench/examples``
folder.

In general, the design flow for using the framework is as follows:

1. Train a network using the train split from a particular dataset.
2. Wrap the network in a ``NeuroBenchModel``.
3. Pass the model, evaluation split dataloader, pre-/post-processors,
   and a list of metrics to the ``Benchmark`` and ``run()``.

Documentation for the framework interfaces can found in
`API.md <API.md>`__.

Development
-----------

If you clone the repo directly for development, poetry can be used to
maintain a virtualenv consistent with a deployment environment. In the
root directory run:

::

   poetry install
   poetry run pytest tests/

Currently the end-to-end examples can be run from the root directory
via:

::

   poetry run python neurobench/examples/dvs_gesture/benchmark.py
   poetry run python neurobench/examples/gsc/benchmark.py
   poetry run python neurobench/examples/mackey_glass/benchmark.py
   poetry run python neurobench/examples/primate_reaching/benchmark.py

The examples may not yet have trained models or a full set of metrics.

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
Anurag Kumar Mishra, Soikat Hasan Ahmed.

Contributing
------------

If you are interested in helping to build this framework, please see the
`contributing guidelines <CONTRIBUTING.rst>`__.

Citation
--------

If you use this framework in your research, please cite the following
whitepaper:

::

   @misc{neurobench_arxiv2023,
         title={NeuroBench: Advancing Neuromorphic Computing through Collaborative, Fair and Representative Benchmarking}, 
         author={Jason Yik and Soikat Hasan Ahmed and Zergham Ahmed and Brian Anderson and Andreas G. Andreou and Chiara Bartolozzi and Arindam Basu and Douwe den Blanken and Petrut Bogdan and Sander Bohte and Younes Bouhadjar and Sonia Buckley and Gert Cauwenberghs and Federico Corradi and Guido de Croon and Andreea Danielescu and Anurag Daram and Mike Davies and Yigit Demirag and Jason Eshraghian and Jeremy Forest and Steve Furber and Michael Furlong and Aditya Gilra and Giacomo Indiveri and Siddharth Joshi and Vedant Karia and Lyes Khacef and James C. Knight and Laura Kriener and Rajkumar Kubendran and Dhireesha Kudithipudi and Gregor Lenz and Rajit Manohar and Christian Mayr and Konstantinos Michmizos and Dylan Muir and Emre Neftci and Thomas Nowotny and Fabrizio Ottati and Ayca Ozcelikkale and Noah Pacik-Nelson and Priyadarshini Panda and Sun Pao-Sheng and Melika Payvand and Christian Pehle and Mihai A. Petrovici and Christoph Posch and Alpha Renner and Yulia Sandamirskaya and Clemens JS Schaefer and André van Schaik and Johannes Schemmel and Catherine Schuman and Jae-sun Seo and Sadique Sheik and Sumit Bam Shrestha and Manolis Sifalakis and Amos Sironi and Kenneth Stewart and Terrence C. Stewart and Philipp Stratmann and Guangzhi Tang and Jonathan Timcheck and Marian Verhelst and Craig M. Vineyard and Bernhard Vogginger and Amirreza Yousefzadeh and Biyan Zhou and Fatima Tuz Zohora and Charlotte Frenkel and Vijay Janapa Reddi},
         year={2023},
         eprint={2304.04640},
         archivePrefix={arXiv},
         primaryClass={cs.AI}
   }
