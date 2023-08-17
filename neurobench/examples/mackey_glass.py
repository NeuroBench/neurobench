import torch

from torch.utils.data import Subset

from neurobench.datasets import MackeyGlass
from neurobench.models import TorchModel
from neurobench.benchmarks import Benchmark

# set seed
np.random.seed(seed=0)

# TODO: still figuring out which should be task (function) parameters
mg = MackeyGlass(17, 0.9)

train_set = Subset(mg, mg.ind_train)
test_set = Subset(mg, mg.ind_test)

## Define model ##
net = nn.Module(...)

## train / load ##
#   net = train(train_set, net)
#   net = torch.load(...)

model = TorchModel(net)

benchmark = Benchmark(model, test_set, [], ["NRMSE", "model_size", "latency", "MACs"]) 
# TODO: metrics?
results = benchmark.run()
print(results)