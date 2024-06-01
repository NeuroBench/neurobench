===================
Membrane Updates
===================

Definition
----------

During the execution of spiking neural networks (SNNs), the average number of updates to the neurons membrane potential is calculated over all neurons in the network, for all timesteps of all tested samples. This metric is specifically designed for spiking neural network implemented with SNNTorch.

The number of membrane updates is calculated by accumulating the number of changes in membrane potential (i.e., the difference between pre- and post-spike membrane potentials) across all neuron layers, timesteps, and input samples, and then normalizing by the number of samples processed.

Implementation Notes
--------------------

When the NeuroBench model is instatiated, certain layers can be recognized automatically for membrane updates calculation.

.. Note::
    The use of this metric are only available for neurons that, when initialized, have the ``init_hidden`` option set to ``True``.

    Example:

    .. code-block::

        import snntorch as snn

        # Initializing a Leaky neuron with init_hidden set to True
        leaky_neuron = snn.Leaky(beta=0.5, output=True, init_hidden=True)



These layers are:

    - snn.SpikingNeuron layers (this is the parent class for all spiking neuron models)

The membrane updates are calculated using hooks that save the pre- and post-spike membrane potentials of the neurons.
At the end of the workload, the total number of membrane potential updates is summed over all layers and batches, and then divided by the number of samples to provide a normalized metric.

