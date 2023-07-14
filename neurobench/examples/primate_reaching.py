from neurobench.datasets import PrimateReaching
from neurobench.models import NeuroBenchModel
from neurobench.benchmarks import Benchmark

# TODO: either in dataloader or in preprocessors need to determine data split
test_set = PrimateReaching(..., split="testing")

# Data preprocessors, eg. outlier segment removal
preprocessors = [Preprocessor(), ...]

## Define model ##
net = ...

# Vincent has ANN, SNN (snntorch), LSTM 

#   model = TorchModel(net)
#   model = SNNTorchModel(net)

# Paul has custom defined SNN

#   class ReachingSNN(TorchModel): 
#       def __init__(self, net):
#           ...
#       def __call__(self, x):
#           ...
#   model = ReachingSNN(net)

metrics = ["R2", "model_size", "latency", "MACs"]

benchmark = Benchmark(model, test_set, preprocessors, metrics)
results = benchmark.run()
print(results)