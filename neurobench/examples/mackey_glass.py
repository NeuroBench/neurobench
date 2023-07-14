import torch

from neurobench.datasets import MackeyGlass
from neurobench.models import NumPyModel
from neurobench.benchmarks import Benchmark

# set seed
np.random.seed(seed=0)

# TODO: need to determine which parameters are intrinsic and user-defined
train_set = MackeyGlass(..., split="training")
test_set = MackeyGlass(..., split="testing")

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