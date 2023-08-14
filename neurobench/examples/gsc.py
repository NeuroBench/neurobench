import torch
import snntorch as snn

from torch import nn
from snntorch import surrogate
from speech2spikes import S2S

from neurobench.datasets import SpeechCommands
from neurobench.models import SNNTorchModel
from neurobench.benchmarks import Benchmark

test_set = SpeechCommands("data/speech_commands/", split="testing")
s2s = S2S()

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

postprocessors = [] # TODO: spike aggregation

static_metrics=["model_size, connection_sparsity, frequency"]
data_metrics=["activation_sparsity, multiply_accumulates, classification_accuracy"]

benchmark = Benchmark(model, test_set, [s2s], postprocessors, [static_metrics, data_metrics])
results = benchmark.run()
print(results)
