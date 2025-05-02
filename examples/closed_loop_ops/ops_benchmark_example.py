import os
import torch
from torch.utils.data import DataLoader, Subset
import sys

sys.path.append("/home/vsun/closed_loop_test/")
from neurobench.envs import OPS, OPSEnv
from neurobench.models.torch_model import TorchModel
from neurobench.benchmarks import BenchmarkClosedLoop

from model import ANNModel

from neurobench.metrics.workload import (
    ActivationSparsity,
    SynapticOperations,
)
from neurobench.metrics.static import (
    Footprint,
    ConnectionSparsity,
)

model_path = "examples/closed_loop_ops/OPS_model_state_dict.pth"
# model_path = "OPS_model_state_dict.pth"

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

time_step = 0.01
num_neurons=96
max_duration = 3.0
min_time_in_target = 0.5 # value in seconds
target_size = 2.5

ops = OPS(
    num_neurons=num_neurons,
    time_step=time_step,
    upper_lmax=100,
    lower_lmax=40,
    upper_lmin=5,
    zero_prob=0.5,
    device=device
)

ops.assign_neurons("examples/closed_loop_ops/neuron1.csv")

env = OPSEnv(
    ops=ops,
    max_duration=max_duration,
    min_time_in_target=min_time_in_target,
    side_radius=10,
    min_distance=8,
    target_size=target_size,
    device=device
)

net = ANNModel(input_dim=num_neurons)
net.load_state_dict(torch.load(model_path, map_location=device))
model = TorchModel(net)

static_metrics = [Footprint, ConnectionSparsity]
workload_metrics = [ActivationSparsity, SynapticOperations]

benchmark = BenchmarkClosedLoop(model, env, [], [], [static_metrics, workload_metrics])

results = benchmark.run(nr_interactions=50, max_length=300, device=device)
print(results)

