import torch
from neurobench.benchmarks.metrics.abstract.workload_metric import AccumulatedMetric
from collections import defaultdict
from .utils import single_layer_MACs


class SynapticOperations(AccumulatedMetric):
    """
    Number of synaptic operations.

    MACs for ANN ACs for SNN

    """

    def __init__(self):

        super().__init__(requires_hooks=True)
        self.MAC = 0
        self.AC = 0
        self.total_synops = 0
        self.total_samples = 0

    def reset(self):
        self.MAC = 0
        self.AC = 0
        self.total_synops = 0
        self.total_samples = 0

    def __call__(self, model, preds, data):
        """
        Multiply-accumulates (MACs) of the model forward.

        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
            inputs: A tensor of model inputs.
        Returns:
            float: Multiply-accumulates.

        """
        for hook in model.connection_hooks:
            inputs = hook.inputs  # copy of the inputs, delete hooks after
            for spikes in inputs:
                # spikes is batch, features, see snntorchmodel wrappper
                # for single_in in spikes:
                if len(spikes) == 1:
                    spikes = spikes[0]
                hook.hook.remove()
                operations, spiking = single_layer_MACs(spikes, hook.layer)
                total_ops, _ = single_layer_MACs(spikes, hook.layer, total=True)
                self.total_synops += total_ops
                if spiking:
                    self.AC += operations
                else:
                    self.MAC += operations
                hook.register_hook()
        # ops_per_sample = ops / data[0].size(0)
        self.total_samples += data[0].size(0)
        return self.compute()

    def compute(self):
        if self.total_samples == 0:
            return {"Effective_MACs": 0, "Effective_ACs": 0, "Dense": 0}
        ac = self.AC / self.total_samples
        mac = self.MAC / self.total_samples
        total_synops = self.total_synops / self.total_samples
        return {"Effective_MACs": mac, "Effective_ACs": ac, "Dense": total_synops}
