import torch

from torch.utils.data import DataLoader

from neurobench.data.datasets import MSWC
from neurobench.models import M5
from neurobench.benchmarks import Benchmark

from torch_mate.data.utils import FewShot

torch.manual_seed(0)

train = lambda net, data: net

model = M5(n_input=48, n_output=40)
dataset = MSWC(root="data/MSWC", subset="evaluation")

few_shot_dataset = FewShot(dataset, 10, 5, 20,
                            incremental=True,
                            cumulative=True)

few_shot_dataloader = DataLoader(few_shot_dataset, batch_size=1)

benchmark = Benchmark(model, dataset, [], ["accuracy", "model_size", "latency", "MACs"])

for session, (X, y) in enumerate(few_shot_dataloader):
    print(f"Session: {session}")

    X_train, X_test = X
    y_train, y_test = y
    
    model = train(model, (X_train[0], y_train[0]))

    session_results = benchmark.run([(X_train[0], y_train[0])])

    print(session_results)
