===================
Connection Sparsity
===================

Definition
----------

Connection sparsity is the proportion of non-zero synaptic weights in a network. It is a measure of the density of connections in a network and can be used to quantify the amount of information that is being transmitted between neurons. For a given model, the connection sparsity is :math:`1 - \sum_l \frac{(m_l)}{(n_l)}`, where :math:`m_l` is the number of nonzero connections and :math:`n_l` is the total potential connections, over each layer :math:`l`. 0 refers to no sparsity (fully connected) and 1 refers to full sparsity (no connections). This metric accounts for deliberate pruning and sparse network architectures. 

Implementation Notes
--------------------