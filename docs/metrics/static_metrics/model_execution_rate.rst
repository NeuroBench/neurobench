====================
Model Execution Rate
====================

Definition
----------
This is not a metric calculated by the harness but is required for official NeuroBench submission. Execution rate, in Hz, of the model computation based on forward inference passes per second, measured in the time-stepped simulation timescale. The time is correlated to real-world data time. For example, if a model is designed to process data from an event camera with 50 ms input stride, the model execution rate is 20 Hz. This metric provides intuition into the deployed real-time responsiveness of a model, as well as its computational requirements.

Note the distinction between *stride* and *bin window* - input can be binned in overlapping windows, but execution rate depends on the temporal stride of window processing. As an example, a model may use 50 ms windows of input and compute every 10 ms, which would give an execution rate of 100 Hz.

Implementation Notes
--------------------
This is not a metric calculated by the harness but is required for official NeuroBench submission.