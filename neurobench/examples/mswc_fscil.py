import torch

from speech2spikes import S2S

from neurobench.datasets import MSWC_FSCIL
from neurobench.models import FSCILModel
from neurobench.benchmarks import FSCILBenchmark

# seed run
torch.manual_seed(0)

# iterable loaders
train_loader = MSWC_FSCIL("data/mswc", split="training")
test_loader = MSWC_FSCIL("data/mswc", split="testing")

s2s = S2S()

## Define model ##
class MAML(FSCILModel):
    def __init__(self, net):
        self.net = net

    def __call__(self, data):
        ...

    def train(self, session, data):
        if session == 0:
            ...
        else:
            ...

    def track_run():
        ...

    def track_batch():
        ...

net = ...
model = MAML(net)

benchmark = FSCILBenchmark(model, test_loader, [s2s], ["accuracy", "model_size", "latency", "MACs"])

for session, train_data in enumerate(train_loader):
    print("Session: {}".format(session))
    model.train(session, train_data)
    session_results = benchmark.run()
    print(session_results)