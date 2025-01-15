import torch

from torch.utils.data import DataLoader
import torchaudio

from neurobench.datasets import SpeechCommands

from neurobench.models import TorchModel
from neurobench.benchmarks import Benchmark

from neurobench.processors.abstract import NeuroBenchPreProcessor, NeuroBenchPostProcessor

from neurobench.metrics.workload import (
    ActivationSparsity,
    SynapticOperations,
    ClassificationAccuracy
)
from neurobench.metrics.static import (
    Footprint,
    ConnectionSparsity,
)

from ANN import M5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data in repo root dir
test_set = SpeechCommands(path="../../data/speech_commands/", subset="testing")

test_set_loader = DataLoader(test_set, batch_size=500, shuffle=True)

net = M5()
net.load_state_dict(torch.load("model_data/m5_ann", map_location=device))

class resample(NeuroBenchPreProcessor):
	def __init__(self):
		self.resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000).to(device)

	def __call__(self, dataset):
		inputs = dataset[0].permute(0, 2, 1)
		inputs = self.resample(inputs)
		return (inputs, dataset[1])

preprocessors = [resample()]

class  convert_to_label(NeuroBenchPostProcessor):

	def __call__(self, output):
		return output.argmax(dim=-1).squeeze()

postprocessors = [convert_to_label()]

## Define model ##
model = TorchModel(net)

static_metrics = [Footprint, ConnectionSparsity]
workload_metrics = [ClassificationAccuracy, ActivationSparsity, SynapticOperations]

benchmark = Benchmark(model, test_set_loader, preprocessors, postprocessors, [static_metrics, workload_metrics])
results = benchmark.run(device=device)
print(results)

# Results:
# {'footprint': 109228, 'connection_sparsity': 0.0,
# 'classification_accuracy': 0.8653339397251905, 'activation_sparsity': 0.3854464619019532, 
# 'synaptic_operations': {'Effective_MACs': 1749994.1556565198, 'Effective_ACs': 0.0, 'Dense': 1902179.0}}