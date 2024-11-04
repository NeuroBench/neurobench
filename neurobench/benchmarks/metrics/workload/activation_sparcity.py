import torch
from neurobench.benchmarks.metrics.base.workload_metric import WorkloadMetric


class ActivationSparsity(WorkloadMetric):
    def __call__(self, model, preds, data):
        """
        Sparsity of model activations.

        Calculated as the number of zero activations over the total number
        of activations, over all layers, timesteps, samples in data.

        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            float: Activation sparsity.

        """
        # TODO: for a spiking model, based on number of spikes over all timesteps over all samples from all layers
        #       Standard FF ANN depends on activation function, ReLU can introduce sparsity.
        total_spike_num, total_neuro_num = 0, 0
        for hook in model.activation_hooks:
            for (
                spikes
            ) in hook.activation_outputs:  # do we need a function rather than a member
                spike_num, neuro_num = torch.count_nonzero(spikes).item(), torch.numel(
                    spikes
                )
                total_spike_num += spike_num
                total_neuro_num += neuro_num

        sparsity = (
            (total_neuro_num - total_spike_num) / total_neuro_num
            if total_neuro_num != 0
            else 0.0
        )
        return sparsity
