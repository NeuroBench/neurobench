from neurobench.metrics.abstract.workload_metric import AccumulatedMetric
from neurobench.metrics.utils.layers import single_layer_MACs


class SynapticOperations(AccumulatedMetric):
    """
    Number of synaptic operations.

    This metric computes the number of Multiply-Accumulate operations (MACs) for
    Artificial Neural Networks (ANN) and Accumulation operations (ACs) for Spiking
    Neural Networks (SNN).

    """

    def __init__(self):
        """Initialize SynapticOperations metric."""

        super().__init__(requires_hooks=True)
        self.MAC = 0
        self.AC = 0
        self.total_synops = 0
        self.total_samples = 0

    def reset(self):
        """
        Reset the metric state for a new evaluation.

        Clears all accumulated values for MAC, AC, synaptic operations, and the total
        number of samples.

        """
        self.MAC = 0
        self.AC = 0
        self.total_synops = 0
        self.total_samples = 0

    def __call__(self, model, preds, data):
        """
        Accumulate the Multiply-Accumulate (MAC) operations or Accumulation (AC)
        operations during the forward pass.

        This method accumulates the operations based on the model's connections, and differentiates between
        ANN (MACs) and SNN (ACs) operations based on the spiking activity.


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
        """
        Compute the average number of operations per sample.

        Returns:
            dict: A dictionary containing:

                - "Effective_MACs": The average MACs per sample.

                - "Effective_ACs": The average ACs per sample.

                - "Dense": The average total synaptic operations per sample.

        """

        if self.total_samples == 0:
            return {"Effective_MACs": 0, "Effective_ACs": 0, "Dense": 0}
        ac = self.AC / self.total_samples
        mac = self.MAC / self.total_samples
        total_synops = self.total_synops / self.total_samples
        return {"Effective_MACs": mac, "Effective_ACs": ac, "Dense": total_synops}
