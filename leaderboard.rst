NeuroBench Leaderboards
=======================

Following are leaderboards for the NeuroBench v1.0 algorithm track benchmarks.

An interactive leaderboard can be found on the webiste `neurobench.ai <https://neurobench.ai>`__.

TODO: table of contents to click to the different headers? Not sure how to do in RST file when it is not compiled by sphinx


+-------------+--------------+-----------+----------------------------+
| Task        | Dataset      | Co        | Task description           |
|             |              | rrectness |                            |
|             |              | metric    |                            |
+=============+==============+===========+============================+
| :ref: `fscil-benchmark` | MSWC         | Accuracy  | Few-shot, continual        |
|             |              |           | learning of keyword        |
|             |              |           | classes                    |
+-------------+--------------+-----------+----------------------------+
| :ref: `event-camera-benchmark`       | Prophesee    | COCO mAP  | Detecting automotive       |
|             | 1MP          |           | objects from event camera  |
|             | Automotive   |           | video                      |
|             |              |           |                            |
+-------------+--------------+-----------+----------------------------+
| :ref: `nhp-motor-benchmark`   | Primate      | R^2       | Predicting fingertip       |
|             | Reaching     |           | velocity from cortical     |
|             |              |           | recordings                 |
+-------------+--------------+-----------+----------------------------+
| :ref: `chaotic-function-benchmark`     | Mackey-Glass | sMAPE     | Autoregressive modeling of |
|             | time series  |           | chaotic functions          |
+-------------+--------------+-----------+----------------------------+

.. _fscil-benchmark:
Keyword Few-Shot Class-Incremental Learning (FSCIL)
---------------------------------------------------

Base accuracy refers to the base class, session 0 test set, while session average refers to the average accuracy across all sessions 0 to 10.

+-----------+-----------+-----------------------------------+-----------+------------------+---------------------+---------------------+--------+--------------------+--------------------+---------------+
| Method    | Reference | Accuracy (Base / Session Average) | Footprint | Model Exec. Rate | Connection Sparsity | Activation Sparsity | Dense  | Eff_MACs           | Eff_ACs            | Date Submitted|
+===========+===========+===================================+===========+==================+=====================+=====================+========+====================+====================+===============+
| M5 ANN    | TODO      | (97.09% / 89.27%)                 | 6.03E6    | 1                | 0.0                 | 0.783               | 2.5E7  | 7.85E6             | 0                  |-              |
+-----------+-----------+-----------------------------------+-----------+------------------+---------------------+---------------------+--------+--------------------+--------------------+---------------+
| SNN       | TODO      | (93.48% / 75.27%)                 | 1.36E7    | 200              | 0.0                 | 0.16                | 3.3E6  | 0                  | 3.65E5             |-              |
+-----------+-----------+-----------------------------------+-----------+------------------+---------------------+---------------------+--------+--------------------+--------------------+---------------+

.. _event-camera-benchmark:
Event Camera Object Detection
-----------------------------

Solution accuracy is measured by COCO mAP.

+------+----------+----------+-------------+------------------+---------------------+---------------------+---------+------------+---------+---------------+
| Rank | Baseline | COCO mAP | Footprint   | Model Exec. Rate | Connection Sparsity | Activation Sparsity | Dense   |Eff_MACs    | Eff_ACs | Date Submitted|
+======+==========+==========+=============+==================+=====================+=====================+=========+============+=========+===============+
| 1    | RED ANN  | 0.42     | .13E7       | 20               | 0.0                 | 0.634               | 2.84E11 | 2.48E11    | 0       | -             |
+------+----------+----------+-------------+------------------+---------------------+---------------------+---------+------------+---------+---------------+
| 2    | Hybrid   | 0.271    | 1.21E7      | 20               | 0.0                 | 0.613               | .85E10  | 3.76E10    | 5.60E8  | -             |
+------+----------+----------+-------------+------------------+---------------------+---------------------+---------+------------+---------+---------------+

.. _nhp-motor-benchmark:
Non-Human Primate Motor Prediction
----------------------------------

For each solution, two sets of metric results are provided, as two
individual solutions are provided per primate that is present in the
dataset.

+------+----------+-------+-------------------+-----------------------+---------------------+---------------------+-------+----------+---------+---------------+
| Rank | Baseline | R^2   | Footprint (bytes) | Model Exec. Rate (Hz) | Connection Sparsity | Activation Sparsity | Dense | Eff_MACs | Eff_ACs | Date Submitted|
+------+----------+-------+-------------------+-----------------------+---------------------+---------------------+-------+----------+---------+---------------+
| 1.   | SNN      | 0.53  | 1648              | 250                   | 0.0                 | 0.7                 | 400   | 0        | 276     | -             |
|      |          | 0.568 | 38848             | 250                   | 0.0                 | 0.                  | 700   | 0        | 551     | -             |
+------+----------+-------+-------------------+-----------------------+---------------------+---------------------+-------+----------+---------+---------------+
| 2.   | ANN      | 0.53  | 20824             | 250                   | 0.0                 | 0.683               | 4704  | 3836     | 0       | -             |
|      |          | 0.558 | 3346              | 250                   | 0.0                 | 0.668               | 7776  | 6103     | 0       | -             |
+------+----------+-------+-------------------+-----------------------+---------------------+---------------------+-------+----------+---------+---------------+

.. _chaotic-function-benchmark:
Chaotic Function Prediction Leaderboard
---------------------------------------

The submitted solutions for the chaotic function prediction task are
evaluated based on the sMAPE metric. Execution rate is not reported as
the data is a synthetic time series, with no real-time correlation.

+------+----------+----------+-----------+------------------+---------------------+---------------------+--------+-----------+---------+---------------+
| Rank | Baseline | Accuracy | Footprint | Model Exec. Rate | Connection Sparsity | Activation Sparsity | Dense  | Eff_MACs  | Eff_ACs | Date Submitted|
+======+==========+==========+===========+==================+=====================+=====================+========+===========+=========+===============+
| 1.   | LSTM     | 13.37    | 4.0E5     | -                | 0.0                 | 0.530               | 6.03E4 | 6.03E4    | 0       | -             |
+------+----------+----------+-----------+------------------+---------------------+---------------------+--------+-----------+---------+---------------+
| 2.   | ESN      | 14.7     | 2.81E5    | -                | 0.876               | 0.0                 | 3.52E4 | 4.37E3    | 0       | -             |
+------+----------+----------+-----------+------------------+---------------------+---------------------+--------+-----------+---------+---------------+