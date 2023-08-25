import torch

from torch.utils.data import Subset, DataLoader

from neurobench.datasets import MackeyGlass
from neurobench.models import TorchModel
from neurobench.benchmarks import Benchmark

from model_data.echo_state_network import EchoStateNetwork

# TODO: still figuring out which should be task (function) parameters
mg = MackeyGlass(tau=17, 
                 constant_past=0.9, 
                 nmg = 10, 
                 beta = 0.2, 
                 gamma = 0.1,
                 dt=1.0, 
                 splits=(8000., 2000.),
                 start_offset=0.,
                 seed_id=0,)

train_set = Subset(mg, mg.ind_train)
test_set = Subset(mg, mg.ind_test)

## Fitting Model ##
seed_id = 0
# TODO: refactor the ESN so that it is correct with the static metrics like model_size
esn = EchoStateNetwork(in_channels=1, 
    reservoir_size = 200, 
    input_scale = torch.tensor([0.2,1],dtype = torch.float64), 
    connect_prob = 0.15, 
    spectral_radius = 1.25, 
    leakage = 0.3, 
    ridge_param = 1.e-8, 
    seed_id = seed_id)
esn.train()
train_data, train_labels = train_set[:]
train_data = train_data.permute(1,0,2) # (batch, timesteps, features)
warmup = 1000
warmup_pts = round(warmup/mg.dt)
train_labels = train_labels[warmup_pts:]
esn.fit(train_data, train_labels, warmup_pts)
torch.save(esn, 'neurobench/examples/model_data/esn.pth')

## Load Model ##
net = torch.load('neurobench/examples/model_data/esn.pth')
test_set_loader = DataLoader(test_set, batch_size=2000, shuffle=False)

model = TorchModel(net)

# static_metrics = ["model_size", "connection_sparsity"]
# data_metrics = ["activation_sparsity", "multiply_accumulates", "MSE"]

static_metrics = ["model_size"]
data_metrics = ["MSE"]

benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, data_metrics]) 
results = benchmark.run()
print(results)

