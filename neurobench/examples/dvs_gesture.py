import torch
from torch.utils.data import DataLoader

import snntorch as snn

from torch import nn
from snntorch import surrogate

from neurobench.datasets import DVSGesture
from neurobench.models import TorchModel
from neurobench.benchmarks import Benchmark

test_set = DVSGesture("data/dvs_gesture/", split="testing", preprocessing="histo_diff")
test_set_loader = DataLoader(test_set, batch_size=16, shuffle=True)

# useless network always returns 1
class OnesNet(nn.Module):
	def __init__(self):
		super(OnesNet, self).__init__()
	
	def forward(self, x):
		return torch.ones(x.shape[0])
net = OnesNet()

# for data in test_set_loader:
# 	print(net(data[0]))
# 	breakpoint()

## Define model ##
model = TorchModel(net)

static_metrics = ["model_size"]
data_metrics = ["classification_accuracy"]

benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, data_metrics])
results = benchmark.run()
print(results)