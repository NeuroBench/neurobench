import torch

from torch.utils.data import DataLoader

from neurobench.datasets import SpeechCommands
from neurobench.preprocessing import S2SProcessor
from neurobench.accumulators import choose_max_count

from neurobench.models import SNNTorchModel
from neurobench.benchmarks import Benchmark

from neurobench.examples.gsc.SNN import net
test_set = SpeechCommands(path="data/speech_commands/", subset="testing")

test_set_loader = DataLoader(test_set, batch_size=500, shuffle=True)

net.load_state_dict(torch.load("neurobench/examples/gsc/model_data/s2s_gsc_snntorch", map_location=torch.device('cpu')))

## Define model ##
model = SNNTorchModel(net)

preprocessors = [S2SProcessor()]
postprocessors = [choose_max_count]

static_metrics = ["model_size", "connection_sparsity"]
data_metrics = ["classification_accuracy", "activation_sparsity"]

benchmark = Benchmark(model, test_set_loader, preprocessors, postprocessors, [static_metrics, data_metrics])
results = benchmark.run()
print(results)

# Results:
# {'model_size': 583900, 'connection_sparsity': 0.0, 
# 'classification_accuracy': 0.8484325349357982, 'activation_sparsity': 0.9674168561138308}