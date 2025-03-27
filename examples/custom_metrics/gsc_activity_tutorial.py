import os
import torch

from torch.utils.data import DataLoader

from neurobench.datasets import SpeechCommands
from neurobench.processors.preprocessors import S2SPreProcessor
from neurobench.processors.postprocessors import ChooseMaxCount

from neurobench.models import SNNTorchModel
from neurobench.benchmarks import Benchmark

from neurobench.metrics.workload import (
    ActivationSparsity,
    SynapticOperations,
    ClassificationAccuracy
)
from neurobench.metrics.static import (
    Footprint,
    ConnectionSparsity,
)
from average_activity_metric import AverageActivityMetric

from examples.gsc.SNN import net
from pprint import pprint
import pathlib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = pathlib.Path(__file__).parent.parent
model_path = os.path.join(file_path, "gsc/model_data/s2s_gsc_snntorch")
data_dir = os.path.join(file_path, "../../data/speech_commands") # data in repo root dir

test_set = SpeechCommands(path=data_dir, subset="testing")

test_set_loader = DataLoader(test_set, batch_size=500, shuffle=True)

net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

## Define model ##
model = SNNTorchModel(net)

preprocessors = [S2SPreProcessor(device=device)]
postprocessors = [ChooseMaxCount()]

static_metrics = [Footprint, ConnectionSparsity]
workload_metrics = [AverageActivityMetric]

benchmark = Benchmark(model, test_set_loader, preprocessors, postprocessors, [static_metrics, workload_metrics])
results = benchmark.run(device=device)
pprint(results)

# plot activity distributions
AverageActivityMetric.plot_activity_distributions(results)

# Results:
# {'Footprint': 583900, 'ConnectionSparsity': 0.0, 
# 'ClassificationAccuracy': 0.85633802969095, 'ActivationSparsity': 0.9668664144456199, 
# 'SynapticOperations': {'Effective_MACs': 0.0, 'Effective_ACs': 3289834.3206724217, 'Dense': 29030400.0}}