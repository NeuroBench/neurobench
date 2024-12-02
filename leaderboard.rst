NeuroBench Leaderboards
=======================

The following are leaderboards for the **NeuroBench v1.0** algorithm track benchmarks, showcasing the performance of various methods across distinct tasks.

The maintained leaderboards cover the following tasks:  
- **[Keyword Few-Shot Class-Incremental Learning (FSCIL)](#fscil-benchmark)**  
- **[Event Camera Object Detection](#event-camera-benchmark)**  
- **[Non-Human Primate Motor Prediction](#nhp-motor-benchmark)**  
- **[Chaotic Function Prediction](#chaotic-function-benchmark)**  

For an interactive version of the leaderboard, visit the official website: `neurobench.ai <https://neurobench.ai>`__.

+------------------------------+------------------+--------------+----------------------------------------------+
| Task                         | Dataset          | Metric       | Task Description                              |
+==============================+==================+==============+==============================================+
| **Keyword FSCIL**            | MSWC             | Accuracy     | Few-shot continual learning of keyword classes |
+------------------------------+------------------+--------------+----------------------------------------------+
| **Event Camera Object        | Prophesee 1MP    | COCO mAP     | Detecting automotive objects from event camera |
| Detection**                  | Automotive       |              | video                                         |
+------------------------------+------------------+--------------+----------------------------------------------+
| **Non-Human Primate Motor    | Primate Reaching | R²           | Predicting fingertip velocity from cortical   |
| Prediction**                 |                  |              | recordings                                    |
+------------------------------+------------------+--------------+----------------------------------------------+
| **Chaotic Function Prediction** | Mackey-Glass  | sMAPE        | Autoregressive modeling of chaotic functions  |
|                              | time series      |              |                                              |
+------------------------------+------------------+--------------+----------------------------------------------+

Each leaderboard highlights key metrics such as accuracy, sparsity, model footprint, and computational efficiency. Below are detailed insights and rankings for each task.

.. _fscil-benchmark:

Keyword Few-Shot Class-Incremental Learning (FSCIL)
---------------------------------------------------

The **Keyword Few-Shot Class-Incremental Learning (FSCIL)** task evaluates models on their ability to perform continual learning in the context of keyword spotting. Keyword spotting involves detecting and recognizing specific spoken words or phrases from audio input—a critical feature for voice-controlled applications.

This benchmark focuses on the challenge of learning new keyword classes incrementally, with limited examples, while retaining performance on previously learned classes. Models are evaluated on two metrics:  
- **Base Accuracy:** Accuracy on the base class (session 0) test set.  
- **Session Average Accuracy:** Average accuracy across all sessions (0 to 10).  

The table below compares methods in terms of accuracy, resource efficiency, and sparsity metrics, offering insights into their trade-offs and performance in this incremental learning scenario.


+-----------+-----------+-----------------------------------+-----------+------------------+---------------------+---------------------+---------+--------------------+--------------------+---------------+
| Method    | Reference | Accuracy (Base / Session Average) | Footprint | Model Exec. Rate | Connection Sparsity | Activation Sparsity | Dense   | Eff_MACs           | Eff_ACs            | Date Submitted|
+===========+===========+===================================+===========+==================+=====================+=====================+=========+====================+====================+===============+
| M5 ANN    | TODO      | (97.09% / 89.27%)                 | 6.03E6    | 1                | 0.0                 | 0.783               | 2.59E7  | 7.85E6             | 0                  |-              |
+-----------+-----------+-----------------------------------+-----------+------------------+---------------------+---------------------+---------+--------------------+--------------------+---------------+
| SNN       | TODO      | (93.48% / 75.27%)                 | 1.36E7    | 200              | 0.0                 | 0.916               | 3.39E6  | 0                  | 3.65E5             |-              |
+-----------+-----------+-----------------------------------+-----------+------------------+---------------------+---------------------+---------+--------------------+--------------------+---------------+

.. _event-camera-benchmark:

Event Camera Object Detection
-----------------------------

**Event Camera Object Detection** evaluates models on their ability to detect and classify objects using data from event-based cameras. Unlike conventional cameras, event cameras capture changes in brightness asynchronously for each pixel, providing high temporal resolution and robustness to motion blur and lighting conditions. These unique properties make them ideal for applications like autonomous driving and robotics.

In this benchmark, models are tasked with detecting automotive objects in event camera video streams. The primary evaluation metric is **COCO mAP (Mean Average Precision)**, which measures detection accuracy. The table below compares methods based on their detection performance, computational efficiency, and sparsity characteristics, highlighting trade-offs relevant for real-world deployments.


+------+----------+----------+-------------+------------------+---------------------+---------------------+---------+------------+---------+---------------+
| Rank | Baseline | COCO mAP | Footprint   | Model Exec. Rate | Connection Sparsity | Activation Sparsity | Dense   |Eff_MACs    | Eff_ACs | Date Submitted|
+======+==========+==========+=============+==================+=====================+=====================+=========+============+=========+===============+
| 1    | RED ANN  | 0.429    | 9.13E7      | 20               | 0.0                 | 0.634               | 2.84E11 | 2.48E11    | 0       | -             |
+------+----------+----------+-------------+------------------+---------------------+---------------------+---------+------------+---------+---------------+
| 2    | Hybrid   | 0.271    | 1.21E7      | 20               | 0.0                 | 0.613               | 9.85E10 | 3.76E10    | 5.60E8  | -             |
+------+----------+----------+-------------+------------------+---------------------+---------------------+---------+------------+---------+---------------+

.. _nhp-motor-benchmark:

Non-Human Primate Motor Prediction
----------------------------------

**Non-Human Primate Motor Prediction** evaluates models on their ability to predict motor behavior, specifically fingertip velocity, from cortical neural recordings. This task is essential for advancing brain-machine interfaces (BMIs), which have applications in neuroprosthetics and understanding motor control mechanisms.

The benchmark provides separate solutions for each primate in the dataset, with models evaluated using the **R² metric**, representing the proportion of variance in the observed data explained by the predicted values. The challenge focuses on achieving high prediction accuracy while maintaining computational efficiency and leveraging sparsity for real-time applications.

The table below presents performance comparisons, including baseline models from the original NeuroBench publication (B-SNN and B-ANN), and highlights improvements made by submitted solutions. Notably, the `tinyRSNN` model demonstrates competitive performance with minimal computational resources, showcasing its potential for lightweight deployment.
 

+------+---------------------------------------------------------------------+-------+-------------------+-----------------------+---------------------+---------------------+---------+----------+---------+---------------+
| Rank | Baseline                                                            | R^2   | Footprint (bytes) | Model Exec. Rate (Hz) | Connection Sparsity | Activation Sparsity | Dense   | Eff_MACs | Eff_ACs | Date Submitted|
+------+---------------------------------------------------------------------+-------+-------------------+-----------------------+---------------------+---------------------+---------+----------+---------+---------------+
| 1.   | `bigSNN <http://1.117.17.41/neural-decoding-grand-challenge/>`__    | 0.698 | 4833360           | 250                   | 0.0                 | 0.968               | 1206272 | 0        | 42003   | 2024-08-02    |
+------+---------------------------------------------------------------------+-------+-------------------+-----------------------+---------------------+---------------------+---------+----------+---------+---------------+
| 2.   | `tinyRSNN <http://1.117.17.41/neural-decoding-grand-challenge/>`__  | 0.66  | 27144             | 250                   | 0.455               | 0.984               | 13440   | 0        | 304     | 2024-08-02    |
+------+---------------------------------------------------------------------+-------+-------------------+-----------------------+---------------------+---------------------+---------+----------+---------+---------------+
| 3.   | B-SNN                                                               | 0.593 | 19648             | 250                   | 0.0                 | 0.997               | 4900    | 0        | 276     | -             |
|      |                                                                     | 0.568 | 38848             | 250                   | 0.0                 | 0.999               | 9700    | 0        | 551     | -             |
+------+---------------------------------------------------------------------+-------+-------------------+-----------------------+---------------------+---------------------+---------+----------+---------+---------------+
| 4.   | B-ANN                                                               | 0.593 | 20824             | 250                   | 0.0                 | 0.683               | 4704    | 3836     | 0       | -             |
|      |                                                                     | 0.558 | 33496             | 250                   | 0.0                 | 0.668               | 7776    | 6103     | 0       | -             |
+------+---------------------------------------------------------------------+-------+-------------------+-----------------------+---------------------+---------------------+---------+----------+---------+---------------+

The results from the `BioCas challenge <http://1.117.17.41/neural-decoding-grand-challenge/>`__ are averaged over all primate datasets. One sees that the R^2 score is higher for the submitted solutions compared to the baselines, with the best solution achieving an R^2 score of 0.698.
Intersetingly, the tinyRSNN model is able to achieve near optimal performance with an extremely small number of operations.


.. _chaotic-function-benchmark:

Chaotic Function Prediction Leaderboard
---------------------------------------

**Chaotic Function Prediction** challenges models to accurately predict values in chaotic time series data, a complex task due to the sensitivity of chaotic systems to initial conditions. This benchmark uses synthetic time series, such as the Mackey-Glass dataset, to evaluate the ability of models to perform autoregressive predictions in highly nonlinear and dynamic environments.

The primary evaluation metric is **sMAPE (Symmetric Mean Absolute Percentage Error)**, which measures prediction accuracy while being robust to scale differences. Since the dataset is synthetic and not tied to real-time scenarios, execution rate is not considered for evaluation.

The table below highlights the performance of various methods, emphasizing their ability to balance accuracy and computational efficiency. This task has implications for modeling in scientific simulations, financial forecasting, and other domains where chaotic systems are prevalent.

+------+----------+----------+-----------+------------------+---------------------+---------------------+--------+-----------+---------+---------------+
| Rank | Baseline | Accuracy | Footprint | Model Exec. Rate | Connection Sparsity | Activation Sparsity | Dense  | Eff_MACs  | Eff_ACs | Date Submitted|
+======+==========+==========+===========+==================+=====================+=====================+========+===========+=========+===============+
| 1.   | LSTM     | 13.37    | 4.90E5    | -                | 0.0                 | 0.530               | 6.03E4 | 6.03E4    | 0       | -             |
+------+----------+----------+-----------+------------------+---------------------+---------------------+--------+-----------+---------+---------------+
| 2.   | ESN      | 14.79     | 2.81E5   | -                | 0.876               | 0.0                 | 3.52E4 | 4.37E3    | 0       | -             |
+------+----------+----------+-----------+------------------+---------------------+---------------------+--------+-----------+---------+---------------+ 
