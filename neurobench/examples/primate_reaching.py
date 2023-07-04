from neurobench.datasets import PrimateReaching
from neurobench.models import NeuroBenchModel
from neurobench.benchmarks import Benchmark

# TODO: either in dataloader or in preprocessors need to determine data split
test_set = PrimateReaching(..., split="testing")

# Data preprocessors, eg. outlier segment removal
preprocessors = [Preprocessor(), ...]

## Define model ##
# Vincent has ANN, SNN (snntorch), LSTM
# Paul has custom defined SNN
class ReachingModel(NeuroBenchModel): 
    def __init__(self, net):
        ...

    def __call__(self, x):
    	...

    def track_run():
        ...

    def track_batch():
        ...

# load model
net = ...
model = ReachingModel(net)

# TODO: for latency do we need cross-correlation or is it always just one timestep to prediction?
metrics = ["R2", "model_size", "latency", "MACs"]

benchmark = Benchmark(model, test_set, preprocessors, metrics)
results = benchmark.run()
print(results)