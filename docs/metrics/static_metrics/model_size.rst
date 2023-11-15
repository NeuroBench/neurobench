===================
Model Size
===================

Definition
----------
A measure of the memory footprint, in bytes, required to represent a model, which reflects quantization, parameters, and buffering requirements. The metric summarizes (and can be further broken down into) synaptic weight count, weight precision, trainable neuron parameters, data buffers, etc. Zero weights are included, as they are addressed in the connection sparsity metric.

Implementation Notes
--------------------