import torch
from neurobench.metrics.abstract.workload_metric import AccumulatedMetric
from collections import defaultdict


class MembraneUpdates(AccumulatedMetric):
    """
    Membrane potential updates metric.

    This metric computes the number of membrane potential updates occurring during the
    forward pass of the model. The updates are tracked per neuron, per layer.

    """

    def __init__(self):
        """Initialize the MembraneUpdates metric."""
        super().__init__(requires_hooks=True)
        self.total_samples = 0
        self.neuron_membrane_updates = defaultdict(int)

    def reset(self):
        """Reset the metric state for a new evaluation."""
        self.total_samples = 0
        self.neuron_membrane_updates = defaultdict(int)

    def __call__(self, model, preds, data):
        """
        Accumulate the number of membrane updates for each model forward pass.

        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            float: Number of membrane potential updates.

        """
        for hook in model.activation_hooks:
            layer_type = hook.layer.__class__.__name__
            updates = self.neuron_membrane_updates[layer_type]

            # Vectorized computation of updates
            if len(hook.pre_fire_mem_potential) > 1:
                pre_fire_mem = torch.stack(hook.pre_fire_mem_potential[1:])
                post_fire_mem = torch.stack(hook.post_fire_mem_potential[1:])
                updates += torch.count_nonzero(pre_fire_mem - post_fire_mem).item()

            # Add the number of elements in the first post_fire_mem_potential
            if hook.post_fire_mem_potential:
                updates += hook.post_fire_mem_potential[0].numel()

            # Update the dictionary
            self.neuron_membrane_updates[layer_type] = updates

        # Increment total_samples
        self.total_samples += data[0].size(0)

        # Return computed results
        return self.compute()

    def compute(self):
        """
        Compute the total membrane updates normalized by the number of samples.

        Returns:
            float: Compute the total updates to each neuron's membrane potential within the model,
            aggregated across all neurons and normalized by the number of samples processed.

        """
        if self.total_samples == 0:
            return 0

        total_mem_updates = sum(self.neuron_membrane_updates.values())

        return total_mem_updates / self.total_samples
