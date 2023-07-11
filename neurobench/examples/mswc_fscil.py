import torch

from speech2spikes import S2S

from neurobench.datasets import MSWC_FSCIL
from neurobench.models import NeuroBenchModel
from neurobench.benchmarks import Benchmark

# seed run
torch.manual_seed(0)

# iterable loaders
train_loader = MSWC_FSCIL("data/mswc", split="training")
test_loader = MSWC_FSCIL("data/mswc", split="testing")

s2s = S2S()

## Define model ##
class MAML(NeuroBenchModel):
    def __init__(self, net):
        self.net = net

    def __call__(self, data):
        ...

    def track_run():
        ...

    def track_batch():
        ...

net = ...
model = MAML(net)

for session, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):
    print("Session: {}".format(session))
    
    ## train model using train_data ##
    net = train(net, train_data)

    ## run benchmark ##
    model = NeuroBenchModel(net)
    benchmark = Benchmark(model, test_data, [s2s], ["accuracy", "model_size", "latency", "MACs"])

    session_results = benchmark.run()
    print(session_results)