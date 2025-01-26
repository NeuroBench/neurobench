import torch
from neurobench.metrics.abstract.workload_metric import WorkloadMetric


class ActivationSparsity(WorkloadMetric):
    """
    Sparsity of model activations.

    Calculated as the number of zero activations over the total number of activations,
    over all layers, timesteps, samples in data.

    """

    def __init__(self):
        """Initialize the ActivationSparsity metric."""

        super().__init__(requires_hooks=True)

    def __call__(self, model, preds, data):
        """
        Compute activation sparsity.

        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            float: Activation sparsity

        """
        # TODO: for a spiking model, based on number of spikes over all timesteps over all samples from all layers
        #       Standard FF ANN depends on activation function, ReLU can introduce sparsity.
        total_spike_num = 0  # Count of non-zero activations
        total_neuro_num = 0  # Count of all activations

        for hook in model.activation_hooks:
            # Skip layers with no outputs
            if not hook.activation_outputs:
                continue

            for (
                spikes
            ) in hook.activation_outputs:  # do we need a function rather than a member
                spike_num, neuro_num = torch.count_nonzero(spikes).item(), torch.numel(
                    spikes
                )
                total_spike_num += spike_num
                total_neuro_num += neuro_num

        # Compute sparsity
        if total_neuro_num == 0:  # Prevent division by zero
            return 0.0

        sparsity = (total_neuro_num - total_spike_num) / total_neuro_num
        return sparsity
