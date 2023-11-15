===================
Synaptic Operations
===================

Definition
----------
Average number of synaptic operations per model execution, based on neuron activations and the associated fanout synapses. This metric is further subdivided into dense, effective multiply-accumulate, and effective accumulate synaptic operations (Dense, Eff_MACs, Eff_ACs). Dense accounts for all zero and nonzero neuron activations and synaptic connections, and reflects the number of operations necessary on hardware that does not support sparsity. Eff_MACs and Eff_ACs only count effective synaptic operations by disregarding zero activations (e.g., produced by the ReLU function in an ANN or no spike in an SNN) and zero connections, thus reflecting operation cost on sparsity-aware hardware. Synaptic operations with non-binary activation are considered multiply-accumulates (MACs), while those with binary activation are considered accumulates (ACs).

Implementation Notes
--------------------