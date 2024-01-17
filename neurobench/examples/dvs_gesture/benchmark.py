import torch
from torch.utils.data import DataLoader

import snntorch as snn

from torch import nn
from snntorch import surrogate

from neurobench.datasets import DVSGesture
from neurobench.models import SNNTorchModel
from neurobench.benchmarks import Benchmark
from neurobench.postprocessing.postprocessor import aggregate,choose_max_count

from neurobench.examples.dvs_gesture.CSNN import Conv_SNN
test_set = DVSGesture("data/dvs_gesture/", split="testing", preprocessing="stack")
test_set_loader = DataLoader(test_set, batch_size=16, shuffle=True,drop_last=True)

net = Conv_SNN()

# The pre-trained model is not available, this demo loads an untrained model.
net.load_state_dict(torch.load('neurobench/examples/dvs_gesture/model_data/DVS_SNN_untrained.pth'))

## Define model ##
model = SNNTorchModel(net)

# postprocessors
postprocessors = [choose_max_count]

static_metrics = ["footprint", "connection_sparsity"]
workload_metrics = ["synaptic_operations", "activation_sparsity", "classification_accuracy"]

benchmark = Benchmark(model, test_set_loader, [], postprocessors, [static_metrics, workload_metrics])
results = benchmark.run()
print(results)