import torch

from torch.utils.data import DataLoader

from neurobench.datasets import SpeechCommands
from neurobench.processors.preprocessors import S2SPreProcessor
from neurobench.processors.postprocessors import ChooseMaxCount

from neurobench.models import SNNTorchModel
from neurobench.benchmarks import Benchmark

from SNN import net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data in repo root dir
test_set = SpeechCommands(path="../../../data/speech_commands/", subset="testing")

test_set_loader = DataLoader(test_set, batch_size=500, shuffle=True)

net.load_state_dict(torch.load("model_data/s2s_gsc_snntorch", map_location=torch.device('cpu')))

## Define model ##
model = SNNTorchModel(net)

preprocessors = [S2SPreProcessor(device=device)]
postprocessors = [ChooseMaxCount()]

static_metrics = ["footprint", "connection_sparsity"]
workload_metrics = ["classification_accuracy", "activation_sparsity", "synaptic_operations"]

benchmark = Benchmark(model, test_set_loader, preprocessors, postprocessors, [static_metrics, workload_metrics])
results = benchmark.run(device=device)
print(results)

# Results:
# {'footprint': 583900, 'connection_sparsity': 0.0,
# 'classification_accuracy': 0.8484325295196562, 'activation_sparsity': 0.9675956131759854, 
# 'synaptic_operations': {'Effective_MACs': 0.0, 'Effective_ACs': 3556689.9895502045, 'Dense': 29336955.0}}