import os
import torch
from snn import Net

# Tonic library is used for DVS Gesture dataset loading and processing
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader

from neurobench.models import SNNTorchModel
from neurobench.processors.postprocessors import ChooseMaxCount
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

file_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(file_path, "model_data/dvs_gesture_snn")
data_dir = os.path.join(file_path, "../../data/dvs_gesture") # data in repo root dir

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

net = Net()
net.load_state_dict(torch.load(model_path, map_location=device))

model = SNNTorchModel(net)

# Load the dataset, here we are using the Tonic library
test_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                     transforms.Downsample(spatial_factor=0.25),
                                     transforms.ToFrame(sensor_size=(32, 32, 2),
                                                        n_time_bins=150),
                                    ])
test_set = tonic.datasets.DVSGesture(save_to=data_dir, transform=test_transform, train=False)
test_set_loader = DataLoader(test_set, batch_size=16,
                         collate_fn=tonic.collation.PadTensors(batch_first=True))

preprocessors = []
postprocessors = [ChooseMaxCount()]

static_metrics = [Footprint, ConnectionSparsity]
workload_metrics = [ClassificationAccuracy, ActivationSparsity, SynapticOperations]

benchmark = Benchmark(model, test_set_loader, preprocessors, postprocessors, [static_metrics, workload_metrics])
results = benchmark.run(device=device)
print(results)

# Results:
# {'Footprint': 304828, 'ConnectionSparsity': 0.0, 
# 'ClassificationAccuracy': 0.8636363636363633, 'ActivationSparsity': 0.9507192967815323, 
# 'SynapticOperations': {'Effective_MACs': 9227011.575757576, 'Effective_ACs': 30564617.685606062, 'Dense': 891206400.0}}