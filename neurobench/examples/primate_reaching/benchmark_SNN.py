import torch
from neurobench.datasets import PrimateReaching
from neurobench.models.snntorch_models import SNNTorchModel
from neurobench.benchmarks import Benchmark

from neurobench.examples.primate_reaching.SNN import SNN
# Download data to /data/primate_reaching/PrimateReachingDataset. See PrimateReaching
# class for download instructions.

# The dataloader and preprocessor has been combined together into a single class
# data/primate_reaching/PrimateReachingDataset/
primate_reaching_dataset = PrimateReaching(file_path="/Users/paul/Downloads/", filename="indy_20170131_02.mat",
                                           num_steps=250, train_ratio=0.8, mode="3D", bin_width=0.004,
                                           biological_delay=50)
test_set = primate_reaching_dataset.create_dataloader(primate_reaching_dataset.ind_test, batch_size=256, shuffle=True)

net = SNN(input_size=primate_reaching_dataset.input_feature_size)
net.load_state_dict(torch.load('model_data/snn.pt', map_location=torch.device('cpu')))

# Give the user the option to load their pretrained weights
# net.load_state_dict(torch.load("neurobench/examples/primate_reaching/model_data/model_parameters.pth"))

model = SNNTorchModel(net)

# metrics = ["r_squared", "model_size", "latency", "MACs"]
static_metrics = ["model_size"]
data_metrics = ["r2", 'MACs', 'latency']

# Benchmark expects the following:
benchmark = Benchmark(model, test_set, [], [], [static_metrics, data_metrics])
results = benchmark.run()
print(results)