import torch
from neurobench.benchmarks.metrics.base.workload_metric import AccumulatedMetric
from collections import defaultdict


class MembraneUpdates(AccumulatedMetric):

    def __init__(self):
        """Init metric state."""
        super().__init__(requires_hooks=True)
        self.total_samples = 0
        self.neuron_membrane_updates = defaultdict(int)

    def reset(self):
        """Reset metric state."""
        self.total_samples = 0
        self.neuron_membrane_updates = defaultdict(int)

    def __call__(self, model, preds, data):
        """
        Number of membrane updates of the model forward.

        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            float: Number of membrane potential updates.

        """
        for hook in model.activation_hooks:
            for index_mem in range(len(hook.pre_fire_mem_potential) - 1):
                pre_fire_mem = hook.pre_fire_mem_potential[index_mem + 1]
                post_fire_mem = hook.post_fire_mem_potential[index_mem + 1]
                nr_updates = torch.count_nonzero(pre_fire_mem - post_fire_mem)
                self.neuron_membrane_updates[str(type(hook.layer))] += int(nr_updates)
            if len(hook.post_fire_mem_potential) > 0:
                self.neuron_membrane_updates[str(type(hook.layer))] += int(
                    torch.numel(hook.post_fire_mem_potential[0])
                )
        self.total_samples += data[0].size(0)
        return self.compute()

    def compute(self):
        """
        Compute membrane updates using accumulated data.

        Returns:
            float: Compute the total updates to each neuron's membrane potential within the model,
            aggregated across all neurons and normalized by the number of samples processed.

        """
        if self.total_samples == 0:
            return 0

        total_mem_updates = 0
        for key in self.neuron_membrane_updates:
            total_mem_updates += self.neuron_membrane_updates[key]

        total_updates_per_sample = total_mem_updates / self.total_samples
        return total_updates_per_sample
