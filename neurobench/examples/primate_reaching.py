from neurobench.datasets import PrimateReaching
from neurobench.models.torch_model import TorchModel
from neurobench.benchmarks import Benchmark

import torch
import torch.nn as nn

# Download data to /data/primate_reaching/PrimateReachingDataset. See PrimateReaching
# class for download instructions.

# The dataloader and preprocessor has been combined together into a single class
primate_reaching_dataset = PrimateReaching(file_path="data/primate_reaching/PrimateReachingDataset/", filename="indy_20170131_02.mat",
                                           num_steps=7, train_ratio=0.8, mode="3D", model_type="ANN")
test_set = primate_reaching_dataset.create_dataloader(primate_reaching_dataset.ind_test, batch_size=256, shuffle=True)

## Define model ##
# The model defined here is a vanilla Fully Connected Network
class ANNModel(nn.Module):
    def __init__(self, input_dim=96, layer1=32, layer2=48, output_dim=2, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm1 = nn.BatchNorm1d(layer1)
        self.batchnorm2 = nn.BatchNorm1d(layer2)
        self.batch_size = None

    def forward(self, x):
        self.batch_size = x.shape[0]

        x = self.activation(self.fc1(x.view(self.batch_size, -1)))
        x = self.batchnorm1(x)
        x = self.activation(self.dropout(self.fc2(x)))
        x = self.batchnorm2(x)
        x = self.fc3(x)

        return x

net = ANNModel(input_dim=primate_reaching_dataset.input_feature_size*primate_reaching_dataset.num_steps,
               layer1=32, layer2=48, output_dim=2, dropout_rate=0.5)

# Give the user the option to load their pretrained weights?
# net.load_state_dict(torch.load("model_state_dict.pth"))

model = TorchModel(net)

# metrics = ["r_squared", "model_size", "latency", "MACs"]
static_metrics = ["model_size"]
data_metrics = ["r2"]

# Benchmark expects the following:
benchmark = Benchmark(model, test_set, [], [], [static_metrics, data_metrics])
results = benchmark.run()
print(results)