===================
Connection Sparsity
===================

Definition
----------

Connection sparsity is the proportion of zero-valued synaptic weights in a network. It is a measure of the density of connections in a network and can be used to quantify the amount of information that is being transmitted between neurons. For a given model, the connection sparsity is :math:`\frac{\sum_l m_l}{\sum_l n_l}`, where :math:`m_l` is the number of zero-valued weights and :math:`n_l` is the total weights, over each layer :math:`l`. 0 refers to no sparsity (fully connected) and 1 refers to full sparsity (no connections). This metric accounts for deliberate pruning and sparse network architectures. 

Implementation Notes
--------------------
The layers that are supported include:
    - Linear
    - Conv1d, Conv2d, Conv3d
    - RNN, RNNBase, RNNCell
    - LSTM, LSTMBase, LSTMCell
    - GRU, GRUBase, GRUCell

Custom connections:
    Your model may contain layers which are involved in operations that implement a connection, such as a matrix multiplication (e.g., torch.matmul), but are not part of the above list.

To ensure that your custom connection layers are tracked by metrics:

1. **Define weights as `nn.Parameter`:** Any tensor used for matrix multiplication or custom operations should be defined as an `nn.Parameter`.
2. **Name with 'weight':** Include the word `weight` in the variable name (e.g., `self.weight1 = torch.nn.Parameter(..)`) so it is recognized as a weight parameter by the tracking system.


We go through the network, extract instances of these layers, count the number of weights and count the number of zero weights.