Neuron Operations Metric
========================

The `NeuronOperations` metric is designed to measure the computational workload associated with neuron activity in spiking neural networks (SNNs). It tracks the number of operations required to update the membrane potential of neurons during the forward pass of a model. These operations include the reset mechanisms defined in the `neuron_ops_reset_operations` dictionary, such as "subtract" and "zero".

Purpose
-------
The metric provides insights into the computational cost of neuron updates, which is critical for analyzing and optimizing the efficiency of spiking neural networks. By understanding the workload associated with different neuron types and reset mechanisms, researchers can identify bottlenecks and improve model performance.

Reset Mechanisms
----------------
Each neuron type has specific reset mechanisms that determine how the membrane potential is updated after a spike. The two main reset mechanisms are:

1. **Subtract**: The membrane potential is reduced by a certain value after a spike.
2. **Zero**: The membrane potential is reset to zero after a spike.

The computational cost of these mechanisms is defined in the `neuron_ops_reset_operations` dictionary. For example, the "Leaky" neuron type has the following costs:
- Subtract mechanism: 4 operations
- Zero mechanism: 4 operations

Why 4 Operations for "Leaky" Neurons?
-------------------------------------
The computational cost of 4 operations for both the "subtract" and "zero" mechanisms in "Leaky" neurons is an abstraction that represents the number of basic mathematical operations required to perform the reset. These operations include the steps involved in updating the membrane potential, checking conditions, and writing the updated state back to memory.

**Subtract Mechanism**
If the `reset_mechanism` is set to "subtract", the membrane potential :math:`U[t+1]` will have the `threshold` subtracted from it whenever the neuron emits a spike. The update equation is:

.. math::

    U[t+1] = \beta U[t] + I_{\rm in}[t+1] - R U_{\rm thr}

Here’s the breakdown of the 4 operations:

1. **Decay Term**: Multiply the previous membrane potential :math:`U[t]` by the decay factor :math:`\beta` (1 operation).
2. **Input Current**: Add the input current :math:`I_{\rm in}[t+1]` to the decayed potential (1 operation).
3. **Reset Multiplication**: Multiply the reset factor :math:`R` by the threshold :math:`U_{\rm thr}` (1 operation).
4. **Threshold Subtraction**: Subtract the result of the reset multiplication from the decayed potential and input current (1 operation).

**Zero Mechanism**
If the `reset_mechanism` is set to "zero", the membrane potential :math:`U[t+1]` will be reset to zero whenever the neuron emits a spike. The update equation is:

.. math::

    U[t+1] = \beta U[t] + I_{\rm syn}[t+1] - R(\beta U[t] + I_{\rm in}[t+1])

Here’s the breakdown of the 4 operations:

1. **Decay Term**: Multiply the previous membrane potential :math:`U[t]` by the decay factor :math:`\beta` (1 operation).
2. **Input Current**: Add the synaptic input current :math:`I_{\rm syn}[t+1]` to the decayed potential (1 operation).
3. **Reset Multiplication**: Multiply the reset factor :math:`R` by the sum of the decayed potential and input current :math:`(\beta U[t] + I_{\rm in}[t+1])` (1 operation).
4. **Reset Subtraction**: Subtract the result of the reset multiplication from the decayed potential and input current (1 operation).

**Why Abstract the Cost to 4 Operations?**

The value of 4 operations is an abstraction that simplifies the computational workload into a consistent metric. While the actual number of operations may vary slightly depending on the implementation, this abstraction provides a way to compare the computational cost of different neuron types and reset mechanisms. It is particularly useful for analyzing and optimizing spiking neural networks across various implementations.

Example: Leaky Neuron with snnTorch
-----------------------------------
Let’s consider an example using the "Leaky" neuron type with `snntorch`. Assume we have a layer of "Leaky" neurons, and we want to compute the workload for the "subtract" and "zero" reset mechanisms.

1. **Subtract Mechanism**:
   
   - After a spike, the membrane potential is reduced by a fixed value.
   - If there are 100 neurons in the layer and each neuron spikes once, the total computational cost is:
    .. math::
        \text{Total Cost} = \text{Number of Neurons} \times \text{Cost per Subtract}
        = 100 \times 4 = 400 \text{ operations.}

2. **Zero Mechanism**:
   
   - After a spike, the membrane potential is reset to zero.
   - If the same 100 neurons spike once, the total computational cost is:
    .. math::
        \text{Total Cost} = \text{Number of Neurons} \times \text{Cost per Zero}
        = 100 \times 4 = 400 \text{ operations.}

Outputs
-------
The `NeuronOperations` metric provides two key outputs:

1. **Effective Neuron Ops**: The total number of operations actually performed by neurons, normalized by the number of samples. This value accounts for the actual activity of the neurons during the forward pass, considering only the updates that occur when neurons spike.

2. **Neuron Dense Ops**: The total number of operations that would be computed if all neurons were updated at every time step, regardless of whether they spiked or not. This represents the theoretical maximum workload for the network under full activity.

These outputs help quantify the computational workload of the network, both in terms of actual activity and theoretical maximum activity, and can be used to optimize the model's efficiency.
