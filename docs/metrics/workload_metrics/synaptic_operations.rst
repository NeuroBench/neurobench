===================
Synaptic Operations
===================

Definition
----------
Average number of synaptic operations per model execution, based on neuron activations and the associated fanout synapses. This metric is further subdivided into dense, effective multiply-accumulate, and effective accumulate synaptic operations (Dense, Eff_MACs, Eff_ACs). 

Dense accounts for all zero and nonzero neuron activations and synaptic connections, and reflects the number of operations necessary on hardware that does not support sparsity. 

Eff_MACs and Eff_ACs only count effective synaptic operations by disregarding zero activations (e.g., produced by the ReLU function in an ANN or no spike in an SNN) and zero connections, thus reflecting operation cost on sparsity-aware hardware. Synaptic operations with non-binary activation are considered multiply-accumulates (MACs), while those with binary activation are considered accumulates (ACs).

Implementation Notes
--------------------
Pooling and normalization are not considered in any metrics currently.

Like the activation sparsity metric, this uses hooks on the Modules. Instead of using hooks on the activation modules, it uses hooks on the connection modules.

The layers that are supported include:
    - Linear
    - Conv1d, Conv2d, Conv3d
    - RNN, RNNBase, RNNCell
    - LSTM, LSTMBase, LSTMCell
    - GRU, GRUBase, GRUCell

This uses pre-hooks, saving the input to each of the layers. At the end of a batch, all the inputs are binarized (for both Effective MACs and ACs).

This is demonstrated below for a linear layer. Assume the bias to be zero, resulting in a forward pass following :math:`y=W \cdot x`. An example weight and activation vector is presented:

.. math::   
    W \cdot x=
    \begin{bmatrix}
        1.3 & 0 & -3.2 \\
        0 & 1.2 & -5.1 \\
        -1.1 & 2.7 & 0 \\
        \end{bmatrix} \cdot \begin{bmatrix}
            0 \\
            1.3 \\
            -1 \\
        \end{bmatrix} 

Both the weight matrix and the activation vector contain zeroes, so effective synaptic operations are only performed for a subset of the weight and input combinations. We make the weight matrix and input matrix binary by replacing non-zeroes with 1. Then, we compute the forward pass to yield effective operations.

.. math::
    \begin{bmatrix}
        1 & 0 & 1 \\
        0 & 1 & 1 \\
        1 & 1 & 0 \\
    \end{bmatrix} \cdot \begin{bmatrix}
        0 \\
        1 \\
        1 \\
    \end{bmatrix} = 
    \begin{bmatrix}
        1 \\
        2 \\
        1 \\
    \end{bmatrix} \rightarrow 
    \text{Effective SynOps} = 4

For Dense synaptic operation, instead of binarizing we convert all elements to 1. The forward pass yields all zero and non-zero operations. In the above example, the number of Dense synaptic operations is 9.

With these augmented input tensors and weights from the hooks we re-run the forward pass. The final number of operations is the average over all of the samples. Samples is defined over the batch dimension (0 dimension).