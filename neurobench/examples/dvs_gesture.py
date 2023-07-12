import torch
import snntorch as snn

from torch import nn
from snntorch import surrogate

from neurobench.datasets import DVSGesture
from neurobench.models import SNNTorchModel
from neurobench.benchmarks import Benchmark

test_set = DVSGesture("data/dvsgesture/", split="testing")

net = ...

## Define model ##
model = SNNTorchModel(net)

benchmark = Benchmark(model, test_set, [s2s], ["accuracy", "model_size", "latency", "MACs"])
results = benchmark.run()
print(results)