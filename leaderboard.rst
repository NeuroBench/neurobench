NeuroBench Leaderboards
=======================

Following are leaderboards for the NeuroBench v1.0 algorithm track benchmarks.

Current tasks for which a leaderboard is maintained are `Keyword Few-Shot Class-Incremental Learning (FSCIL) <#fscil-benchmark>`__, `Event Camera Object Detection <#event-camera-benchmark>`__, `Non-Human Primate Motor Prediction <#nhp-motor-benchmark>`__ and `Chaotic Function Prediction Leaderboard <#chaotic-function-benchmark>`__.


An interactive leaderboard can be found on the webiste `neurobench.ai <https://neurobench.ai>`__.

TODO: table of contents to click to the different headers? Not sure how to do in RST file when it is not compiled by sphinx


+-------------+--------------+-----------+----------------------------+
| Task        | Dataset      | Co        | Task description           |
|             |              | rrectness |                            |
|             |              | metric    |                            |
+=============+==============+===========+============================+
| Keyword     | MSWC         | Accuracy  | Few-shot, continual        |
| FSCIL       |              |           | learning of keyword        |
|             |              |           | classes                    |
+-------------+--------------+-----------+----------------------------+
| Event       | Prophesee    | COCO mAP  | Detecting automotive       |
| Camera      | 1MP          |           | objects from event camera  |
| Object      | Automotive   |           | video                      |
| Detection   |              |           |                            |
+-------------+--------------+-----------+----------------------------+
| NHP Motor   | Primate      | R^2       | Predicting fingertip       |
| Prediction  | Reaching     |           | velocity from cortical     |
|             |              |           | recordings                 |
+-------------+--------------+-----------+----------------------------+
| Chaotic     | Mackey-Glass | sMAPE     | Autoregressive modeling of |
| Function    | time series  |           | chaotic functions          |
+-------------+--------------+-----------+----------------------------+

.. _fscil-benchmark:
Keyword Few-Shot Class-Incremental Learning (FSCIL)
---------------------------------------------------

Base accuracy refers to the base class, session 0 test set, while session average refers to the average accuracy across all sessions 0 to 10.

+-----------+-----------+-----------------------------------+-----------+------------------+---------------------+---------------------+--------+--------------------+--------------------+---------------+
| Method    | Reference | Accuracy (Base / Session Average) | Footprint | Model Exec. Rate | Connection Sparsity | Activation Sparsity | Dense  | Eff_MACs           | Eff_ACs            | Date Submitted|
+===========+===========+===================================+===========+==================+=====================+=====================+========+====================+====================+===============+
| M5 ANN    | TODO      | (97.09% / 89.27%)                 | 6.03E6    | 1                | 0.0                 | 0.783               | 2.59E7  | 7.85E6             | 0                  |-              |
+-----------+-----------+-----------------------------------+-----------+------------------+---------------------+---------------------+--------+--------------------+--------------------+---------------+
| SNN       | TODO      | (93.48% / 75.27%)                 | 1.36E7    | 200              | 0.0                 | 0.916                | 3.39E6  | 0                  | 3.65E5             |-              |
+-----------+-----------+-----------------------------------+-----------+------------------+---------------------+---------------------+--------+--------------------+--------------------+---------------+

.. _event-camera-benchmark:
Event Camera Object Detection
-----------------------------

Solution accuracy is measured by COCO mAP.

+------+----------+----------+-------------+------------------+---------------------+---------------------+---------+------------+---------+---------------+
| Rank | Baseline | COCO mAP | Footprint   | Model Exec. Rate | Connection Sparsity | Activation Sparsity | Dense   |Eff_MACs    | Eff_ACs | Date Submitted|
+======+==========+==========+=============+==================+=====================+=====================+=========+============+=========+===============+
| 1    | RED ANN  | 0.429     | 9.13E7       | 20               | 0.0                 | 0.634               | 2.84E11 | 2.48E11    | 0       | -             |
+------+----------+----------+-------------+------------------+---------------------+---------------------+---------+------------+---------+---------------+
| 2    | Hybrid   | 0.271    | 1.21E7      | 20               | 0.0                 | 0.613               | 9.85E10  | 3.76E10    | 5.60E8  | -             |
+------+----------+----------+-------------+------------------+---------------------+---------------------+---------+------------+---------+---------------+

.. _nhp-motor-benchmark:
Non-Human Primate Motor Prediction
----------------------------------

For each solution, two sets of metric results are provided, as an
individual solution is provided per primate that is present in the
dataset. B-SNN and B-ANN refer to the results in the original NeuroBench publication, which serve as a baseline to the submitted solutions. 

+------+----------+-------+-------------------+-----------------------+---------------------+---------------------+-------+----------+---------+---------------+
| Rank | Baseline | R^2   | Footprint (bytes) | Model Exec. Rate (Hz) | Connection Sparsity | Activation Sparsity | Dense | Eff_MACs | Eff_ACs | Date Submitted|
+------+----------+-------+-------------------+-----------------------+---------------------+---------------------+-------+----------+---------+---------------+
| 1.   | 'bigSNN <http://1.117.17.41/neural-decoding-grand-challenge/>'__ | 0.698  | 4833360              | 250                   | 0.0                 | 0.968                 | 1206272   | 0        | 42003     | 2024-08-02             |
+------+----------+-------+-------------------+-----------------------+---------------------+---------------------+-------+----------+---------+---------------+
| 2.   | 'tinyRSNN <http://1.117.17.41/neural-decoding-grand-challenge/>'__ | 0.66  | 27144              | 250                   | 0.455                 | 0.984                 | 13440   | 0        | 304     | 2024-08-02             |
+------+----------+-------+-------------------+-----------------------+---------------------+---------------------+-------+----------+---------+---------------+
| 3.   | B-SNN    | 0.593  | 19648              | 250                   | 0.0                 | 0.997                 | 4900   | 0        | 276     | -             |
|      |          | 0.568 | 38848             | 250                   | 0.0                 | 0.999                  | 9700   | 0        | 551     | -             |
+------+----------+-------+-------------------+-----------------------+---------------------+---------------------+-------+----------+---------+---------------+
| 4.   | B-ANN      | 0.593  | 20824             | 250                   | 0.0                 | 0.683               | 4704  | 3836     | 0       | -             |
|      |          | 0.558 | 33496              | 250                   | 0.0                 | 0.668               | 7776  | 6103     | 0       | -             |
+------+----------+-------+-------------------+-----------------------+---------------------+---------------------+-------+----------+---------+---------------+

The results from the 'BioCas challenge <http://1.117.17.41/neural-decoding-grand-challenge/>' are averaged over all primate datasets. One sees that the R^2 score is higher for the submitted solutions compared to the baselines, with the best solution achieving an R^2 score of 0.698.
Intersetingly, the tinyRSNN model is able to achieve near optimal performance with an extremely small number of operations.


.. _chaotic-function-benchmark:
Chaotic Function Prediction Leaderboard
---------------------------------------

The submitted solutions for the chaotic function prediction task are
evaluated based on the sMAPE metric. Execution rate is not reported as
the data is a synthetic time series, with no real-time correlation.

+------+----------+----------+-----------+------------------+---------------------+---------------------+--------+-----------+---------+---------------+
| Rank | Baseline | Accuracy | Footprint | Model Exec. Rate | Connection Sparsity | Activation Sparsity | Dense  | Eff_MACs  | Eff_ACs | Date Submitted|
+======+==========+==========+===========+==================+=====================+=====================+========+===========+=========+===============+
| 1.   | LSTM     | 13.37    | 4.90E5     | -                | 0.0                 | 0.530               | 6.03E4 | 6.03E4    | 0       | -             |
+------+----------+----------+-----------+------------------+---------------------+---------------------+--------+-----------+---------+---------------+
| 2.   | ESN      | 14.79     | 2.81E5    | -                | 0.876               | 0.0                 | 3.52E4 | 4.37E3    | 0       | -             |
+------+----------+----------+-----------+------------------+---------------------+---------------------+--------+-----------+---------+---------------+