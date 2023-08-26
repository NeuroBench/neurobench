#
# NOTE: This task is still under development.
#

import torch

from speech2spikes import S2S

from neurobench.datasets import MSWC_FSCIL
from neurobench.models import NeuroBenchModel
from neurobench.benchmarks import Benchmark

# seed run
torch.manual_seed(0)

# TODO: discuss general FSCIL dataloader in subgroup

# iterable loaders
#   train_loader = MSWC_FSCIL("data/mswc", split="training")
#   test_loader = MSWC_FSCIL("data/mswc", split="testing")

# generalized format
#   mswc = MSWC()
#   train_loader, test_loader = FSCIL_Loader(mswc, classes=[[base_classes],[ses1],[ses2],...], max_train_samples=..., max_test_samples=...)

s2s = S2S()

## Define model ##
class SNN_MAML(TorchModel):
    def __init__(self, net):
        self.net = net

    def __call__(self, data):
        ...

net = ...
model = SNN_MAML(net)
benchmark = Benchmark(model, test_data, [s2s], ["accuracy", "model_size", "latency", "MACs"])

for session, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):
    print("Session: {}".format(session))
    
    ## train model using train_data ##
    net = train(net, train_data)

    ## run benchmark ##
    session_results = benchmark.run() # TODO: need to re-instantiate model/benchmark when net changes?
    print(session_results)