import torch
from neurobench.metrics.abstract.workload_metric import AccumulatedMetric
from collections import defaultdict


neuron_ops_reset_operations = {
    "Leaky": {
        "subtract": 4,
        "zero": 4,
    },
    "Synaptic": {
        "subtract": 6,
        "zero": 8,
    },
    "Lapicque": {
        "subtract": 11,
        "zero": 11,
    },
    "Alpha": {
        "subtract": 18,
        "zero": 24,
    },
}
"""
The `neuron_ops_reset_operations` dictionary defines the computational cost associated
with resetting the membrane potential of neurons for different neuron types. The reset
mechanisms are categorized into two types:

1. **Subtract**: Represents a reset mechanism where the membrane potential is reduced
   by a certain value. The value associated with this mechanism indicates the computational
   cost (in terms of basic operations) required to perform this type of reset.

2. **Zero**: Represents a reset mechanism where the membrane potential is reset to zero.
   The value associated with this mechanism indicates the computational cost (in terms of
   basic operations) required to perform this type of reset.

### Neuron Types and Their Computational Costs:
- **Leaky**:
  - Subtract mechanism: 4 operations
  - Zero mechanism: 4 operations
- **Synaptic**:
  - Subtract mechanism: 6 operations
  - Zero mechanism: 8 operations
- **Lapicque**:
  - Subtract mechanism: 11 operations
  - Zero mechanism: 11 operations
- **Alpha**:
  - Subtract mechanism: 18 operations
  - Zero mechanism: 24 operations

### Purpose:
The values in this dictionary represent the computational cost (measured in terms of
basic operations like addition, subtraction, etc.) required for each neuron type to
reset its membrane potential using a specific reset mechanism.
"""


class NeuronOperations(AccumulatedMetric):
    """
    Neuron operations metric.

    This metric computes the number of operations performed by neurons during the
    forward pass of the model. The operations are tracked per neuron, per layer.

    The `NeuronOperations` metric is designed to measure the computational workload
    associated with neuron activity in spiking neural networks. Specifically, it tracks
    the number of operations required to update the membrane potential of neurons during
    the forward pass. These operations include the reset mechanisms defined in the
    `neuron_ops_reset_operations` dictionary, such as "subtract" and "zero".

    """

    def __init__(self):
        """Initialize the NeuronOperations metric."""
        super().__init__(requires_hooks=True)
        self.total_samples = 0
        self.dense = defaultdict(int)
        self.macs = defaultdict(int)

    def reset(self):
        """Reset the metric state for a new evaluation."""
        self.total_samples = 0
        self.dense = defaultdict(int)
        self.macs = defaultdict(int)

    def __call__(self, model, preds, data):
        """
        Accumulate the neuron operations.

        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            float: Number of membrane potential updates.

        """
        for hook in model.activation_hooks:
            layer_type = hook.layer.__class__.__name__
            reset_mechanism = hook.layer._reset_mechanism
            updates = 0

            if len(hook.pre_fire_mem_potential) > 1:
                pre_fire_mem = torch.stack(hook.pre_fire_mem_potential[1:])
                post_fire_mem = torch.stack(hook.post_fire_mem_potential[1:])
                updates += torch.count_nonzero(pre_fire_mem - post_fire_mem).item()
            if hook.post_fire_mem_potential:
                updates += hook.post_fire_mem_potential[0].numel()

            self.macs[layer_type] += (
                updates * neuron_ops_reset_operations[layer_type][reset_mechanism]
            )
            self.dense[layer_type] += (
                hook.post_fire_mem_potential[0].numel()
                * len(hook.post_fire_mem_potential)
                * neuron_ops_reset_operations[layer_type][reset_mechanism]
            )

        self.total_samples += data[0].size(0)

        return self.compute()

    def compute(self):
        """
        Compute the total membrane updates normalized by the number of samples.

        Returns:
            float: Compute the total updates to each neuron's membrane potential within the model,
            aggregated across all neurons and normalized by the number of samples processed.

        """
        if self.total_samples == 0:
            return {"Effective Neuron Ops": 0, "Neuron Dense Ops": 0}

        macs = sum(self.macs.values())
        dense = sum(self.dense.values())

        return {
            "Effective Neuron Ops": macs / self.total_samples,
            "Neuron Dense Ops": dense / self.total_samples,
        }
