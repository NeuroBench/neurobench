from neurobench.metrics.abstract.workload_metric import AccumulatedMetric
from collections import defaultdict
import torch


class ActivationSparsityByLayer(AccumulatedMetric):

    def __init__(self):
        self.layer_sparsity = defaultdict(float)
        self.layer_neuro_num = defaultdict(int)
        self.layer_spike_num = defaultdict(int)

    def reset(self):
        self.layer_sparsity = defaultdict(float)
        self.layer_neuro_num = defaultdict(int)
        self.layer_spike_num = defaultdict(int)

    def __call__(self, model, preds, data):

        for hook in model.activation_hooks:
            name = hook.name
            if name is None:
                continue
            for output in hook.activation_outputs:
                spike_num, neuro_num = torch.count_nonzero(
                    output.dequantize() if output.is_quantized else output
                ).item(), torch.numel(output)

                self.layer_spike_num[name] += spike_num
                self.layer_neuro_num[name] += neuro_num

        return self.compute()

    def compute(self):
        for key in self.layer_neuro_num:
            sparsity = (
                (self.layer_neuro_num[key] - self.layer_spike_num[key])
                / self.layer_neuro_num[key]
                if self.layer_neuro_num[key] != 0
                else 0.0
            )
            self.layer_sparsity[key] = sparsity
        return dict(self.layer_sparsity)
