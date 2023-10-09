import torch

from torch.utils.data import DataLoader
import torchaudio

from neurobench.datasets import SpeechCommands

from neurobench.models import TorchModel
from neurobench.benchmarks import Benchmark

from neurobench.preprocessing import NeuroBenchProcessor

from ANN import M5

test_set = SpeechCommands(path="data/speech_commands/", subset="testing")

test_set_loader = DataLoader(test_set, batch_size=500, shuffle=True)

net = M5()
net.load_state_dict(torch.load("neurobench/examples/gsc/model_data/m5_ann", map_location=torch.device('cpu')))

class resample(NeuroBenchProcessor):
	def __init__(self):
		self.resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)

	def __call__(self, dataset):
		inputs = dataset[0].permute(0, 2, 1)
		inputs = self.resample(inputs)
		return (inputs, dataset[1])

preprocessors = [resample()]

def convert_to_label(output):
	return output.argmax(dim=-1).squeeze()

postprocessors = [convert_to_label]

## Define model ##
model = TorchModel(net)

static_metrics = ["model_size", "connection_sparsity"]
data_metrics = ["classification_accuracy", "activation_sparsity", "synaptic_operations"]

benchmark = Benchmark(model, test_set_loader, preprocessors, postprocessors, [static_metrics, data_metrics])
results = benchmark.run()
print(results)

# Results:
# {'model_size': 109228, 'connection_sparsity': 0.0, 
# 'classification_accuracy': 0.8653339397251905, 'activation_sparsity': 0.3854464619019532, 
# 'synaptic_operations': {'Effective_MACs': 1749994.1556565198, 'Effective_ACs': 0.0, 'Dense': 1902179.0}}