from neurobench.benchmarks.metrics.base import StaticMetric


class ParameterCount(StaticMetric):
    def __call__(self, model):
        return sum(p.numel() for p in model.__net__().parameters())