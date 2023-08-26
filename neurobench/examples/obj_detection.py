#
# NOTE: This task is still under development.
#

from neurobench.datasets import Prophesee1MP
from neurobench.models import NeuroBenchModel
from neurobench.benchmarks import Benchmark

test_set = Prophesee1MP(..., split="testing")

# Data preprocessors, wrap metavision
preprocessors = [Preprocessor(), ...]

## Define model ##
class ObjDetectionModel(NeuroBenchModel):
    def __init__(self, net, box_coder, head):
        ...

    def __call__(self, x):
    	...

# load model
net = ...
box_coder = ...
head = ...
model = ObjDetectionModel(net, box_coder, head)

# metric implementations should also wrap metavision
metrics = ["mAP", "model_size", "detection_latency", "MACs"]

# TODO: anything different for batched test set for user side / in Benchmark?
benchmark = Benchmark(model, test_set, preprocessors, metrics)
results = benchmark.run()
print(results)
