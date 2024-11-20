import torch
from neurobench.benchmarks.metrics.abstract.workload_metric import WorkloadMetric


class ActivationSparsity(WorkloadMetric):
    def __init__(self):
        """Init metric state."""
        super().__init__(requires_hooks=True)

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
        total_spike_num = 0  # Count of non-zero activations
        total_neuro_num = 0  # Count of all activations

        for hook in model.activation_hooks:
            # Skip layers with no outputs
            if not hook.activation_outputs:
                continue

            # Concatenate activations for efficient processing
            all_activations = torch.cat(hook.activation_outputs, dim=0)

            # Count non-zero and total elements using batched operations
            total_spike_num += all_activations.count_nonzero().item()
            total_neuro_num += all_activations.numel()

        # Compute sparsity
        if total_neuro_num == 0:  # Prevent division by zero
            return 0.0

        sparsity = (total_neuro_num - total_spike_num) / total_neuro_num
        return sparsity
