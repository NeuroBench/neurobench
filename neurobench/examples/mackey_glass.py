import numpy as np

from neurobench.datasets import MackeyGlass
from neurobench.models import NumPyModel
from neurobench.benchmarks import Benchmark

# set seed
np.random.seed(seed=0)

# TODO: need to determine which parameters are intrinsic and user-defined
train_set = MackeyGlass(..., split="training")
test_set = MackeyGlass(..., split="testing")

# ???? How to generalize NumPyModel
class ESNNumPyModel(NumPyModel):
    ...

model = ESNNumPyModel()
benchmark = Benchmark(model, test_set, [], ["NRMSE", "model_size", "latency", "MACs"]) 
# TODO: latency calculation different than the reaching task?
results = benchmark.run()
print(results)