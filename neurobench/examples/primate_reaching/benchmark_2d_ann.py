import torch
from torch.utils.data import DataLoader, Subset

from neurobench.datasets import PrimateReaching
from neurobench.models.torch_model import TorchModel
from neurobench.benchmarks import Benchmark

from neurobench.examples.primate_reaching.ANN import ANNModel2D
# Download data to /data/primate_reaching/PrimateReachingDataset. See PrimateReaching
# class for download instructions.

# The dataloader and preprocessor has been combined together into a single class
dataset = PrimateReaching(file_path="data/primate_reaching/PrimateReachingDataset/", filename="indy_20170131_02.mat",
                          num_steps=1, train_ratio=0.5, bin_width=0.004,
                          biological_delay=0)

test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=256, shuffle=False)

net = ANNModel2D(input_dim=dataset.input_feature_size, layer1=32, layer2=48, 
               output_dim=2, bin_window=0.2, dropout_rate=0.5)

# Give the user the option to load their pretrained weights
# TODO: currently model is not trained
# net.load_state_dict(torch.load("neurobench/examples/primate_reaching/model_data/2d_model_parameters"))

model = TorchModel(net)

static_metrics = ["model_size"]
data_metrics = ["r2"]

# Benchmark expects the following:
benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, data_metrics])
results = benchmark.run()
print(results)