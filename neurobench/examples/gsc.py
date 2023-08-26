import torch
import snntorch as snn

from torch.utils.data import DataLoader

from torch import nn
from snntorch import surrogate

from neurobench.datasets import SpeechCommands
from neurobench.preprocessing import S2SProcessor
from neurobench.accumulators import choose_max_count

from neurobench.models import SNNTorchModel
from neurobench.benchmarks import Benchmark

test_set = SpeechCommands(path="data/speech_commands/", subset="testing")

test_set_loader = DataLoader(test_set, batch_size=500, shuffle=True)

beta = 0.9
spike_grad = surrogate.fast_sigmoid()
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(20, 256),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(256, 256),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(256, 256),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(256, 35),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
)
net.load_state_dict(torch.load("neurobench/examples/model_data/s2s_gsc_snntorch", map_location=torch.device('cpu')))

## Define model ##
model = SNNTorchModel(net)

preprocessors = [S2SProcessor()]
postprocessors = [choose_max_count]

# static_metrics=["model_size", "connection_sparsity", "frequency"]
# data_metrics=["activation_sparsity", "multiply_accumulates", "classification_accuracy"]

static_metrics = ["model_size"]
data_metrics = ["classification_accuracy"]

benchmark = Benchmark(model, test_set_loader, preprocessors, postprocessors, [static_metrics, data_metrics])
results = benchmark.run()
print(results)
