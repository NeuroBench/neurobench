import torch
from torch.utils.data import DataLoader

import snntorch as snn

from torch import nn
from snntorch import surrogate

from neurobench.datasets import DVSGesture
from neurobench.models import SNNTorchModel
from neurobench.benchmarks import Benchmark
from neurobench.accumulators.accumulator import aggregate,choose_max_count

from neurobench.examples.model_data.ConvSNN import Conv_SNN
test_set = DVSGesture("data/dvs_gesture/", split="testing", preprocessing="stack")
test_set_loader = DataLoader(test_set, batch_size=16, shuffle=True,drop_last=True)

net = Conv_SNN()
net.load_state_dict(torch.load('neurobench/examples/model_data/DVS_SNN_untrained.pth'))

## Define model ##
model = SNNTorchModel(net)

# postprocessors
postprocessors = [choose_max_count]

static_metrics = ["model_size"]
data_metrics = ["classification_accuracy"]

benchmark = Benchmark(model, test_set_loader, [], postprocessors, [static_metrics, data_metrics])
results = benchmark.run()
print(results)