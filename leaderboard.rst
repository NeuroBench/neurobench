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

+---+---+-------+----+-------+--------+--------+----+-----+---+-----+
| R | B | Acc   | F  | Model | Conn   | Acti   | D  | Ef  | E | D   |
| a | a | uracy | oo | Exec. | ection | vation | en | f_M | f | ate |
| n | s |       | tp | Rate  | Sp     | Sp     | se | ACs | f | Sub |
| k | e |       | ri |       | arsity | arsity |    |     | _ | mit |
|   | l |       | nt |       |        |        |    |     | A | ted |
|   | i |       |    |       |        |        |    |     | C |     |
|   | n |       |    |       |        |        |    |     | s |     |
|   | e |       |    |       |        |        |    |     |   |     |
+===+===+=======+====+=======+========+========+====+=====+===+=====+
| 1 | R | 0.429 | :m | 20    | 0.0    | 0.634  | :  | :ma | 0 | D   |
|   | E |       | at |       |        |        | ma | th: |   | ATE |
|   | D |       | h: |       |        |        | th | `2. |   |     |
|   | A |       | `9 |       |        |        | :` | 48  |   |     |
|   | N |       | .1 |       |        |        | 2. | \ti |   |     |
|   | N |       | 3  |       |        |        | 84 | mes |   |     |
|   |   |       | \t |       |        |        |  \ |  10 |   |     |
|   |   |       | im |       |        |        | ti | ^{1 |   |     |
|   |   |       | es |       |        |        | me | 1}` |   |     |
|   |   |       |  1 |       |        |        | s  |     |   |     |
|   |   |       | 0^ |       |        |        | 10 |     |   |     |
|   |   |       | 7` |       |        |        | ^{ |     |   |     |
|   |   |       |    |       |        |        | 11 |     |   |     |
|   |   |       |    |       |        |        | }` |     |   |     |
+---+---+-------+----+-------+--------+--------+----+-----+---+-----+
| 2 | H | 0.271 | :m | 20    | 0.0    | 0.613  | :  | :m  | : |     |
|   | y |       | at |       |        |        | ma | ath | m |     |
|   | b |       | h: |       |        |        | th | :`3 | a |     |
|   | r |       | `1 |       |        |        | :` | .76 | t |     |
|   | i |       | .2 |       |        |        | 9. | \ti | h |     |
|   | d |       | 1  |       |        |        | 85 | mes | : |     |
|   |   |       | \t |       |        |        |  \ |  10 | ` |     |
|   |   |       | im |       |        |        | ti | ^{1 | 5 |     |
|   |   |       | es |       |        |        | me | 0}` | . |     |
|   |   |       |  1 |       |        |        | s  |     | 6 |     |
|   |   |       | 0^ |       |        |        | 10 |     | 0 |     |
|   |   |       | 7` |       |        |        | ^{ |     | \ |     |
|   |   |       |    |       |        |        | 10 |     | t |     |
|   |   |       |    |       |        |        | }` |     | i |     |
|   |   |       |    |       |        |        |    |     | m |     |
|   |   |       |    |       |        |        |    |     | e |     |
|   |   |       |    |       |        |        |    |     | s |     |
|   |   |       |    |       |        |        |    |     | 1 |     |
|   |   |       |    |       |        |        |    |     | 0 |     |
|   |   |       |    |       |        |        |    |     | ^ |     |
|   |   |       |    |       |        |        |    |     | 8 |     |
|   |   |       |    |       |        |        |    |     | ` |     |
+---+---+-------+----+-------+--------+--------+----+-----+---+-----+

.. raw:: html

   <!-- Interactive HTML table:
   <!DOCTYPE html>
   <html>
   <head>
     <title>Sortable Table</title>
     <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.css">
     <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
     <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.js"></script>
   </head>
   <body>

   <table id="sortable-table">
     <thead>
       <tr>
         <th>Baseline</th>
         <th>mAP</th>
         <th>Footprint (bytes)</th>
         <th>Model Exec. Rate (Hz)</th>
         <th>Connection Sparsity</th>
         <th>Activation Sparsity</th>
         <th>SynOps (Dense)</th>
         <th>SynOps (Eff_MACs)</th>
         <th>SynOps (Eff_ACs)</th>
       </tr>
     </thead>
     <tbody>
       <tr>
         <td>RED ANN</td>
         <td>0.429</td>
         <td>$9.13 \times 10^7$</td>
         <td>20</td>
         <td>0.0</td>
         <td>0.634</td>
         <td>$2.84 \times 10^{11}$</td>
         <td>$2.48 \times 10^{11}$</td>
         <td>0</td>
       </tr>
       <tr>
         <td>Hybrid</td>
         <td>0.271</td>
         <td>$1.21 \times 10^7$</td>
         <td>20</td>
         <td>0.0</td>
         <td>0.613</td>
         <td>$9.85 \times 10^{10}$</td>
         <td>$3.76\times 10^{10}$</td>
         <td>$5.60\times10^8$</td>
       </tr>
     </tbody>
   </table>

   <script>
     $(document).ready( function () {
         $('#sortable-table').DataTable();
     } );
   </script>

   </body>
   </html>
    -->

Chaotic Function Prediction Leaderboard
---------------------------------------

The submitted solutions for the chaotic function prediction task are
evaluated based on the sMAPE metric. Execution rate is not reported as
the data is a synthetic time series, with no real-time correlation.

+---+---+-------+----+-------+--------+--------+----+-----+---+-----+
| R | B | Acc   | F  | Model | Conn   | Acti   | D  | Ef  | E | D   |
| a | a | uracy | oo | Exec. | ection | vation | en | f_M | f | ate |
| n | s |       | tp | Rate  | Sp     | Sp     | se | ACs | f | Sub |
| k | e |       | ri |       | arsity | arsity |    |     | _ | mit |
|   | l |       | nt |       |        |        |    |     | A | ted |
|   | i |       |    |       |        |        |    |     | C |     |
|   | n |       |    |       |        |        |    |     | s |     |
|   | e |       |    |       |        |        |    |     |   |     |
+===+===+=======+====+=======+========+========+====+=====+===+=====+
| 1 | L | 13.37 | :m | -     | 0.0    | 0.530  | :m | :   | 0 |     |
| . | S |       | at |       |        |        | at | mat |   |     |
|   | T |       | h: |       |        |        | h: | h:` |   |     |
|   | M |       | `4 |       |        |        | `6 | 6.0 |   |     |
|   |   |       | .9 |       |        |        | .0 | 3\t |   |     |
|   |   |       | 0  |       |        |        | 3  | ime |   |     |
|   |   |       | \t |       |        |        | \t | s 1 |   |     |
|   |   |       | im |       |        |        | im | 0^{ |   |     |
|   |   |       | es |       |        |        | es | 4}` |   |     |
|   |   |       |  1 |       |        |        |  1 |     |   |     |
|   |   |       | 0^ |       |        |        | 0^ |     |   |     |
|   |   |       | 5` |       |        |        | {4 |     |   |     |
|   |   |       |    |       |        |        | }` |     |   |     |
+---+---+-------+----+-------+--------+--------+----+-----+---+-----+
| 2 | E | 14.79 | :m | -     | 0.876  | 0.0    | :m | :m  | 0 |     |
| . | S |       | at |       |        |        | at | ath |   |     |
|   | N |       | h: |       |        |        | h: | :`4 |   |     |
|   |   |       | `2 |       |        |        | `3 | .37 |   |     |
|   |   |       | .8 |       |        |        | .5 |  \t |   |     |
|   |   |       | 1  |       |        |        | 2  | ime |   |     |
|   |   |       | \t |       |        |        | \t | s 1 |   |     |
|   |   |       | im |       |        |        | im | 0^{ |   |     |
|   |   |       | es |       |        |        | es | 3}` |   |     |
|   |   |       |  1 |       |        |        |  1 |     |   |     |
|   |   |       | 0^ |       |        |        | 0^ |     |   |     |
|   |   |       | 5` |       |        |        | {4 |     |   |     |
|   |   |       |    |       |        |        | }` |     |   |     |
+---+---+-------+----+-------+--------+--------+----+-----+---+-----+

Keyword Few-Shot Learning Leaderboard
-------------------------------------

The submitted solutions for the keyword few-shot class-incremental
learning task are evaluated based on the accuracy metric. Base accuracy
refers to accuracy on the 100 base classes after pre-training while
session average accuracy is the average accuracy over all sessions for
the corresponding prototypical baseline.

+---+---+-------+----+-------+--------+--------+----+-----+---+-----+
| R | B | Acc   | F  | Model | Conn   | Acti   | D  | Ef  | E | D   |
| a | a | uracy | oo | Exec. | ection | vation | en | f_M | f | ate |
| n | s | (base | tp | Rate  | Sp     | Sp     | se | ACs | f | Sub |
| k | e | /     | ri |       | arsity | arsity |    |     | _ | mit |
|   | l | se    | nt |       |        |        |    |     | A | ted |
|   | i | ssion |    |       |        |        |    |     | C |     |
|   | n | ave   |    |       |        |        |    |     | s |     |
|   | e | rage) |    |       |        |        |    |     |   |     |
+===+===+=======+====+=======+========+========+====+=====+===+=====+
| 1 | M | (9    | :m | 1     | 0.0    | 0.783  | :m | :ma | 0 | Apr |
| . | 5 | 7.09% | at |       |        |        | at | th: |   | 21  |
|   | A | /     | h: |       |        |        | h: | `7. |   | 2   |
|   | N | 89    | `6 |       |        |        | `2 | 85  |   | 024 |
|   | N | .27%) | .0 |       |        |        | .5 | \ti |   |     |
|   |   |       | 3  |       |        |        | 9  | mes |   |     |
|   |   |       | \t |       |        |        | \t |  10 |   |     |
|   |   |       | im |       |        |        | im | ^6` |   |     |
|   |   |       | es |       |        |        | es |     |   |     |
|   |   |       |  1 |       |        |        |  1 |     |   |     |
|   |   |       | 0^ |       |        |        | 0^ |     |   |     |
|   |   |       | 6` |       |        |        | 7` |     |   |     |
+---+---+-------+----+-------+--------+--------+----+-----+---+-----+
| 2 | S | (9    | :m | 200   | 0.0    | 0.916  | :m | 0   | : | Apr |
| . | N | 3.48% | at |       |        |        | at |     | m | 21  |
|   | N | /     | h: |       |        |        | h: |     | a | 2   |
|   |   | 75    | `1 |       |        |        | `3 |     | t | 024 |
|   |   | .27%) | .3 |       |        |        | .3 |     | h |     |
|   |   |       | 6  |       |        |        | 9  |     | : |     |
|   |   |       | \t |       |        |        | \t |     | ` |     |
|   |   |       | im |       |        |        | im |     | 3 |     |
|   |   |       | es |       |        |        | es |     | . |     |
|   |   |       |  1 |       |        |        |  1 |     | 6 |     |
|   |   |       | 0^ |       |        |        | 0^ |     | 5 |     |
|   |   |       | 7` |       |        |        | 6` |     |   |     |
|   |   |       |    |       |        |        |    |     | \ |     |
|   |   |       |    |       |        |        |    |     | t |     |
|   |   |       |    |       |        |        |    |     | i |     |
|   |   |       |    |       |        |        |    |     | m |     |
|   |   |       |    |       |        |        |    |     | e |     |
|   |   |       |    |       |        |        |    |     | s |     |
|   |   |       |    |       |        |        |    |     |   |     |
|   |   |       |    |       |        |        |    |     | 1 |     |
|   |   |       |    |       |        |        |    |     | 0 |     |
|   |   |       |    |       |        |        |    |     | ^ |     |
|   |   |       |    |       |        |        |    |     | 5 |     |
|   |   |       |    |       |        |        |    |     | ` |     |
+---+---+-------+----+-------+--------+--------+----+-----+---+-----+

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

+---+----+---+---+---------+-----------+----------+----------+---+----+---+
| R | Ba | : | F | Model   | C         | Ac       | Dense    | E | E  | D |
| a | se | m | o | Exec.   | onnection | tivation |          | f | ff | a |
| n | li | a | o | Rate    | Sparsity  | Sparsity |          | f | _A | t |
| k | ne | t | t | (Hz)    |           |          |          | _ | Cs | e |
|   |    | h | p |         |           |          |          | M |    | S |
|   |    | : | r |         |           |          |          | A |    | u |
|   |    | ` | i |         |           |          |          | C |    | b |
|   |    | R | n |         |           |          |          | s |    | m |
|   |    | ^ | t |         |           |          |          |   |    | i |
|   |    | 2 | ( |         |           |          |          |   |    | t |
|   |    | ` | b |         |           |          |          |   |    | t |
|   |    |   | y |         |           |          |          |   |    | e |
|   |    |   | t |         |           |          |          |   |    | d |
|   |    |   | e |         |           |          |          |   |    |   |
|   |    |   | s |         |           |          |          |   |    |   |
|   |    |   | ) |         |           |          |          |   |    |   |
+===+====+===+===+=========+===========+==========+==========+===+====+===+
| 1 | S  | 0 | 1 | 250     | 0.0       | 0.997    | 4900     | 0 | 2  |   |
| . | NN | . | 9 |         |           |          |          |   | 76 |   |
|   |    | 5 | 6 |         |           |          |          |   |    |   |
|   |    | 9 | 4 |         |           |          |          |   |    |   |
|   |    | 3 | 8 |         |           |          |          |   |    |   |
+---+----+---+---+---------+-----------+----------+----------+---+----+---+
|   |    | 0 | 3 | 250     | 0.0       | 0.999    | 9700     | 0 | 5  |   |
|   |    | . | 8 |         |           |          |          |   | 51 |   |
|   |    | 5 | 8 |         |           |          |          |   |    |   |
|   |    | 6 | 4 |         |           |          |          |   |    |   |
|   |    | 8 | 8 |         |           |          |          |   |    |   |
+---+----+---+---+---------+-----------+----------+----------+---+----+---+
| 2 | A  | 0 | 2 | 250     | 0.0       | 0.683    | 4704     | 3 | 0  |   |
| . | NN | . | 0 |         |           |          |          | 8 |    |   |
|   |    | 5 | 8 |         |           |          |          | 3 |    |   |
|   |    | 9 | 2 |         |           |          |          | 6 |    |   |
|   |    | 3 | 4 |         |           |          |          |   |    |   |
+---+----+---+---+---------+-----------+----------+----------+---+----+---+
|   |    | 0 | 3 | 250     | 0.0       | 0.668    | 7776     | 6 | 0  |   |
|   |    | . | 3 |         |           |          |          | 1 |    |   |
|   |    | 5 | 4 |         |           |          |          | 0 |    |   |
|   |    | 5 | 9 |         |           |          |          | 3 |    |   |
|   |    | 8 | 6 |         |           |          |          |   |    |   |
+---+----+---+---+---------+-----------+----------+----------+---+----+---+
