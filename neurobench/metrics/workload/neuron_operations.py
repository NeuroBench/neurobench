import torch
from neurobench.metrics.abstract.workload_metric import AccumulatedMetric
from collections import defaultdict


neuron_macs_reset_operations = {
    "Leaky": {
        "subtract": 2,
        "zero": 2,
    },
    "Synaptic": {
        "subtract": 3,
        "zero": 2,
    },
    "Lapicque": {
        "subtract": 3,
        "zero": 2,
    },
    "Alpha": {
        "subtract": 3,
        "zero": 3,
    },
}


class NeuronOperations(AccumulatedMetric):
    """
    Neuron operations metric.

    This metric computes the number of operations performed by neurons during the
    forward pass of the model. The operations are tracked per neuron, per layer.

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
                updates * neuron_macs_reset_operations[layer_type][reset_mechanism]
            )
            self.dense[layer_type] += (
                hook.post_fire_mem_potential[0].numel()
                * len(hook.post_fire_mem_potential)
                * neuron_macs_reset_operations[layer_type][reset_mechanism]
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
            return {"Neuron MACs": 0, "Neuron Dense": 0}

        macs = sum(self.macs.values())
        dense = sum(self.dense.values())

        return {
            "Effective Neuron MACs": macs / self.total_samples,
            "Neuron Dense": dense / self.total_samples,
        }
