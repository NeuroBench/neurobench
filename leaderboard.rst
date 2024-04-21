NeuroBench Leaderboard
======================

The NeuroBench Leaderboard showcases the best performing solutions for
each task in the NeuroBench harness. It provides a comprehensive
overview of the top-performing algorithms and models submitted by
participants. This leaderboard serves as a benchmark for evaluating the
performance of different approaches and encourages healthy competition
among participants.

Each task in the NeuroBench harness is evaluated based on the NeuroBench
metrics. The leaderboard presents the top solutions based on their
performance scores, allowing researchers and developers to identify the
most effective techniques for various tasks, relevant to the
neuromorphic community.

By regularly updating the leaderboard, the NeuroBench project aims to
foster collaboration, knowledge sharing, and innovation in the field of
neuromorphics. It provides a platform for researchers and practitioners
to showcase their advancements and contribute to the collective
understanding of developing neuromorphic solutions.

Whether you are a participant, researcher, or enthusiast, the NeuroBench
Leaderboard offers valuable insights into the state-of-the-art solutions
and advancements in neuromorphic research. Explore the leaderboard to
discover the best performing solutions and stay up-to-date with the
latest developments in the field.

Let the competition begin!

   **Note:** Interactive leaderboards can be found
   `here <https://neurobench.ai>`__. ## Tasks Overview

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

Event Camera Object Detection Leaderboard
-----------------------------------------

The submitted solutions for the chaotic function prediction task are
evaluated based on the COCO mAP metric.

+------+----------+----------+-------------------+------------------+---------------------+---------------------+------------+-------------+---------+---------------+
| Rank | Baseline | Accuracy | Footprint         | Model Exec. Rate | Connection Sparsity | Activation Sparsity | Dense      | Eff_MACs    | Eff_ACs | Date Submitted|
+======+==========+==========+===================+==================+=====================+=====================+============+=============+=========+===============+
| 1    | RED ANN  | 0.429    | $9.13 \times 10^7$| 20               | 0.0                 | 0.634               | $2.84 \times 10^{11}$ | $2.48 \times 10^{11}$ | 0       | DATE          |
+------+----------+----------+-------------------+------------------+---------------------+---------------------+------------+-------------+---------+---------------+
| 2    | Hybrid   | 0.271    | $1.21 \times 10^7$| 20               | 0.0                 | 0.613               | $9.85 \times 10^{10}$ | $3.76 \times 10^{10}$ | $5.60 \times 10^8$ |               |
+------+----------+----------+-------------------+------------------+---------------------+---------------------+------------+-------------+---------+---------------+

Chaotic Function Prediction Leaderboard
---------------------------------------

The submitted solutions for the chaotic function prediction task are
evaluated based on the sMAPE metric. Execution rate is not reported as
the data is a synthetic time series, with no real-time correlation.

+------+----------+----------+-------------------+------------------+---------------------+---------------------+------------+-------------+---------+---------------+
| Rank | Baseline | Accuracy | Footprint         | Model Exec. Rate| Connection Sparsity | Activation Sparsity | Dense      | Eff_MACs    | Eff_ACs | Date Submitted|
+======+==========+==========+===================+==================+=====================+=====================+============+=============+=========+===============+
| 1.   | LSTM     | 13.37    | $4.90 \times 10^5$| -                | 0.0                 | 0.530               | $6.03 \times 10^{4}$ | $6.03 \times 10^{4}$ | 0       |               |
+------+----------+----------+-------------------+------------------+---------------------+---------------------+------------+-------------+---------+---------------+
| 2.   | ESN      | 14.79    | $2.81 \times 10^5$| -                | 0.876               | 0.0                 | $3.52 \times 10^{4}$ | $4.37 \times 10^{3}$ | 0       |               |
+------+----------+----------+-------------------+------------------+---------------------+---------------------+------------+-------------+---------+---------------+


Keyword Few-Shot Learning Leaderboard
-------------------------------------

The submitted solutions for the keyword few-shot class-incremental
learning task are evaluated based on the accuracy metric. Base accuracy
refers to accuracy on the 100 base classes after pre-training while
session average accuracy is the average accuracy over all sessions for
the corresponding prototypical baseline.

+------+----------+-----------------------------------+--------------------+------------------+---------------------+---------------------+--------------------+--------------------+--------------------+---------------+
| Rank | Baseline | Accuracy (base / session average) | Footprint          | Model Exec. Rate | Connection Sparsity | Activation Sparsity | Dense              | Eff_MACs           | Eff_ACs            | Date Submitted|
+======+==========+===================================+====================+==================+=====================+=====================+====================+====================+====================+===============+
| 1.   | M5 ANN   | (97.09% / 89.27%)                 | $6.03 \times 10^6$ | 1                | 0.0                 | 0.783               | $2.59 \times 10^7$ | $7.85 \times 10^6$ | 0                  | Apr 21 2024   |
+------+----------+-----------------------------------+--------------------+------------------+---------------------+---------------------+--------------------+--------------------+--------------------+---------------+
| 2.   | SNN      | (93.48% / 75.27%)                 | $1.36 \times 10^7$ | 200              | 0.0                 | 0.916               | $3.39 \times 10^6$ | 0                  | $3.65 \times 10^5$ | Apr 21 2024   |
+------+----------+-----------------------------------+--------------------+------------------+---------------------+---------------------+--------------------+--------------------+--------------------+---------------+


Non-Human Primate Prediction
----------------------------

The submitted solutions for the NHP prediction task are evaluated based
on the :math:`R^2` metric. The leaderboard showcases the top-performing
solutions for this task, providing insights into the most effective
approaches for predicting fingertip velocity from cortical recordings in
non-human primates.

For each solution, two sets of metric results are provided, as two
individual solutions are provided per primate that is present in the
dataset.

+------+----------+-------+-------------------+-----------------------+---------------------+---------------------+-------+----------+---------+---------------+
| Rank | Baseline | $R^2$ | Footprint (bytes) | Model Exec. Rate (Hz) | Connection Sparsity | Activation Sparsity | Dense | Eff_MACs | Eff_ACs | Date Submitted|
+------+----------+-------+-------------------+-----------------------+---------------------+---------------------+-------+----------+---------+---------------+
| 1.   | SNN      | 0.593 | 19648             | 250                   | 0.0                 | 0.997               | 4900  | 0        | 276     |               |
|      |          | 0.568 | 38848             | 250                   | 0.0                 | 0.999               | 9700  | 0        | 551     |               |
| 2.   | ANN      | 0.593 | 20824             | 250                   | 0.0                 | 0.683               | 4704  | 3836     | 0       |               |
|      |          | 0.558 | 33496             | 250                   | 0.0                 | 0.668               | 7776  | 6103     | 0       |               |
+------+----------+-------+-------------------+-----------------------+---------------------+---------------------+-------+----------+---------+---------------+
