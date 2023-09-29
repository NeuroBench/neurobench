import torch
from torch.utils.data import DataLoader, Subset

from neurobench.datasets import PrimateReaching
from neurobench.models.snntorch_models import SNNTorchModel
from neurobench.benchmarks import Benchmark
from neurobench.examples.primate_reaching.SNN_3 import SNNModel3

# Download data to /data/primate_reaching/PrimateReachingDataset. See PrimateReaching
# class for download instructions.

# The dataloader and preprocessor has been combined together into a single class
dataset = PrimateReaching(file_path="data/primate_reaching/PrimateReachingDataset/", filename="indy_20170131_02.mat",
                          num_steps=1, train_ratio=0.8, bin_width=0.004,
                          biological_delay=50)
test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=256, shuffle=False)

net = SNNModel3(input_size=dataset.input_feature_size)
net.load_state_dict(torch.load('model_data/snn.pt', map_location=torch.device('cpu'))['model_state_dict'])

# Give the user the option to load their pretrained weights
# TODO: currently model is not trained
# net.load_state_dict(torch.load("neurobench/examples/primate_reaching/model_data/snn3_model_parameters"))

model = SNNTorchModel(net)

# metrics = ["r_squared", "model_size", "latency", "MACs"]
static_metrics = ["model_size"]
data_metrics = ["r2"]

# Benchmark expects the following:
benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, data_metrics])
results = benchmark.run()
print(results)
