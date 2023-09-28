import torch
from torch.utils.data import DataLoader, Subset

from neurobench.datasets import PrimateReaching
from neurobench.models.torch_model import TorchModel
from neurobench.benchmarks import Benchmark

from neurobench.examples.primate_reaching.ANN import ANNModel
# Download data to /data/primate_reaching/PrimateReachingDataset. See PrimateReaching
# class for download instructions.

# The dataloader and preprocessor has been combined together into a single class
dataset = PrimateReaching(file_path="data/primate_reaching/PrimateReachingDataset/", filename="indy_20170131_02.mat",
                                           num_steps=7, train_ratio=0.8)

test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=256, shuffle=True)


net = ANNModel(input_dim=dataset.input_feature_size*dataset.num_steps,
               layer1=32, layer2=48, output_dim=2, dropout_rate=0.5)

# Give the user the option to load their pretrained weights
# TODO: currently model is not trained
# net.load_state_dict(torch.load("neurobench/examples/primate_reaching/model_data/model_parameters"))

model = TorchModel(net)

static_metrics = ["model_size", "connection_sparsity"]
data_metrics = ["r2", "activation_sparsity", "synaptic_operations"]

# Benchmark expects the following:
benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, data_metrics])
results = benchmark.run()
print(results)