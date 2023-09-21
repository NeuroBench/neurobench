from neurobench.datasets import PrimateReaching
from neurobench.models.torch_model import TorchModel
from neurobench.benchmarks import Benchmark

from neurobench.examples.primate_reaching.ANN import ANNModel
# Download data to /data/primate_reaching/PrimateReachingDataset. See PrimateReaching
# class for download instructions.

# The dataloader and preprocessor has been combined together into a single class
primate_reaching_dataset = PrimateReaching(file_path="data/primate_reaching/PrimateReachingDataset/", filename="indy_20170131_02.mat",
                                           num_steps=7, train_ratio=0.8, mode="3D", model_type="ANN")
test_set = primate_reaching_dataset.create_dataloader(primate_reaching_dataset.ind_test, batch_size=256, shuffle=True)



net = ANNModel(input_dim=primate_reaching_dataset.input_feature_size*primate_reaching_dataset.num_steps,
               layer1=32, layer2=48, output_dim=2, dropout_rate=0.5)

# Give the user the option to load their pretrained weights
# net.load_state_dict(torch.load("neurobench/examples/primate_reaching/model_data/model_parameters.pth"))

model = TorchModel(net)

# metrics = ["r_squared", "model_size", "latency", "MACs"]
static_metrics = ["model_size"]
data_metrics = ["r2", "activation_sparsity"]

# Benchmark expects the following:
benchmark = Benchmark(model, test_set, [], [], [static_metrics, data_metrics])
results = benchmark.run()
print(results)