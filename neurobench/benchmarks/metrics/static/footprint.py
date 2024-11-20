from neurobench.benchmarks.metrics.abstract import StaticMetric


class Footprint(StaticMetric):
    def __call__(self, model):
        param_size = 0
        for param in model.__net__().parameters():
            param_size += param.numel() * param.element_size()

        buffer_size = 0
        for buffer in model.__net__().buffers():
            buffer_size += buffer.numel() * buffer.element_size()

        return param_size + buffer_size
