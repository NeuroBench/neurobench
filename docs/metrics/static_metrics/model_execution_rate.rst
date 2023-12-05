====================
Model Execution Rate
====================

Definition
----------
This is not a metric calculated by the harness but is required for official NeuroBench submission. Execution rate, in Hz, of the model computation based on forward inference passes per second, measured in the time-stepped simulation timescale. The time is correlated to real-world data time. For example, if a model is designed to processes data from an event camera with 50 ms input stride, the model execution rate is 20 Hz. This metric provides intuition into the deployed real-time responsiveness of a model, as well as its computational requirements.

Implementation Notes
--------------------
This is not a metric calculated by the harness but is required for official NeuroBench submission.