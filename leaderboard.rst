=======================
NeuroBench Leaderboard
=======================

The NeuroBench Leaderboard showcases the best performing solutions for each task in the NeuroBench harness. It provides a comprehensive overview of the top-performing algorithms and models submitted by participants. This leaderboard serves as a benchmark for evaluating the performance of different approaches and encourages healthy competition among participants.

Each task in the NeuroBench harness is evaluated based on the NeuroBench metrics. The leaderboard presents the top solutions based on their performance scores, allowing researchers and developers to identify the most effective techniques for various tasks, relevant to the neuromorphic community.

By regularly updating the leaderboard, the NeuroBench project aims to foster collaboration, knowledge sharing, and innovation in the field of neuromorphics. It provides a platform for researchers and practitioners to showcase their advancements and contribute to the collective understanding of developing neuromorphic solutions.

Whether you are a participant, researcher, or enthusiast, the NeuroBench Leaderboard offers valuable insights into the state-of-the-art solutions and advancements in neuromorphic research. Explore the leaderboard to discover the best performing solutions and stay up-to-date with the latest developments in the field.

Let the competition begin!

=======================
Tasks overview
=======================
.. table:: NeuroBench algorithm track v1.0 benchmarks.
   :align: center
   :widths: 20% 20% 20% 40%

   +--------------------------+---------------------------------------+-------------------+-----------------------------------------------+
   | Task                     | Dataset                               | Correctness metric| Task description                             |
   +==========================+=======================================+===================+===============================================+
   | Keyword FSCIL            | MSWC                                  | Accuracy          | Few-shot, continual learning of keyword      |
   |                          |                                       |                   | classes.                                      |
   +--------------------------+---------------------------------------+-------------------+-----------------------------------------------+
   | Event Camera Object      | Prophesee 1MP Automotive| COCO mAP       | Detecting automotive objects from event      |
   | Detection                |                                       |                   | camera video.                                 |
   +--------------------------+---------------------------------------+-------------------+-----------------------------------------------+
   | NHP Motor Prediction     | Primate Reaching                      | R^2               | Predicting fingertip velocity from cortical  |
   |                          |                                       |                   | recordings.                                   |
   +--------------------------+---------------------------------------+-------------------+-----------------------------------------------+
   | Chaotic Function         | Mackey-Glass time series              | sMAPE| Autoregressive modeling of chaotic           |
   | Prediction               |                                       |                   | functions.                                    |
   +--------------------------+---------------------------------------+-------------------+-----------------------------------------------+

=======================
Event Camera Object Detection Leaderboard
=======================
Rank  |  Model Name  |  Task  |  Score  |  Date Submitted


=======================
Mackey-Glass Chaotic Function Prediction Leaderboard
=======================
Rank  |  Model Name  |  Task  |  Score  |  Date Submitted

=======================
MSWC Keyword Few-Shot Learning Leaderboard
=======================
Rank  |  Model Name  |  Task  |  Score  |  Date Submitted
.. table:: Baseline results for the keyword few-shot class-incremental learning task. Base accuracy refers to accuracy on the 100 base classes after pre-training while session average accuracy is the average accuracy over all sessions for the corresponding prototypical baseline. The detailed accuracy per session for the different baselines are shown in Figure~\ref{fig:mswc_acc_per_session}.
   :align: center

   +--------------+-------------------------------+------------------+---------------------+---------------------+---------------------+----------------------+---------------------------+------------+
   | Baseline     | Accuracy                      | Footprint        | Model Exec. Rate    | Connection Sparsity | Activation Sparsity | Dense                | Eff_MACs                  | Eff_ACs    |
   +==============+===============================+==================+=====================+=====================+=====================+======================+===========================+============+
   | M5 ANN       | (97.09% / 89.27%)            | $6.03 \times 10^6$ | 1                   | 0.0                 | 0.783               | $2.59 \times 10^7$  | $7.85 \times 10^6$        | 0          |
   +--------------+-------------------------------+------------------+---------------------+---------------------+---------------------+----------------------+---------------------------+------------+
   | SNN          | (93.48% / 75.27%)            | $1.36 \times 10^7$ | 200                 | 0.0                 | 0.916               | $3.39 \times 10^6$  | 0                         | $3.65 \times 10^5$ |
   +--------------+-------------------------------+------------------+---------------------+---------------------+---------------------+----------------------+---------------------------+------------+

=======================
Primate Motor Prediction Leaderboard
=======================
Rank  |  Model Name  |  Task  |  Score  |  Date Submitted