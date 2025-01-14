import torch
from snn import Net

# Tonic library is used for DVS Gesture dataset loading and processing
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader

from neurobench.models import SNNTorchModel
from neurobench.processors.postprocessors import ChooseMaxCount
from neurobench.benchmarks import Benchmark

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

net = Net()
net.load_state_dict(torch.load("model_data/dvs_gesture_snn", map_location=device))

model = SNNTorchModel(net)

# Load the dataset, here we are using the Tonic library
data_dir = "../../../data/dvs_gesture" # data in repo root dir
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

static_metrics = ["footprint", "connection_sparsity"]
workload_metrics = ["classification_accuracy", "activation_sparsity", "synaptic_operations"]

benchmark = Benchmark(model, test_set_loader, preprocessors, postprocessors, [static_metrics, workload_metrics])
results = benchmark.run(device=device)
print(results)

# Results:
# {'footprint': 304828, 'connection_sparsity': 0.0, 
# 'classification_accuracy': 0.8636363636363633, 'activation_sparsity': 0.9507192967815323, 
# 'synaptic_operations': {'Effective_MACs': 9227011.575757576, 'Effective_ACs': 30564577.174242426, 'Dense': 891206400.0}}