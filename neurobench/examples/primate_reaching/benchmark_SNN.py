import torch
from torch.utils.data import DataLoader, Subset

from neurobench.datasets import PrimateReaching
from neurobench.models.snntorch_models import SNNTorchModel
from neurobench.benchmarks import Benchmark
from neurobench.examples.primate_reaching.SNN import SNN

# Download data to /data/primate_reaching/PrimateReachingDataset. See PrimateReaching
# class for download instructions.

# The dataloader and preprocessor has been combined together into a single class
files = ["indy_20160622_01", "indy_20160630_01", "indy_20170131_02",
            "loco_20170210_03", "loco_20170217_02", "loco_20170301_05"]

for filename in files:
    dataset = PrimateReaching(file_path="data/primate_reaching/PrimateReachingDataset/", filename=filename,
                              num_steps=50, train_ratio=0.5, bin_width=0.004,
                              biological_delay=0, max_segment_length=False)
    test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=len(dataset.ind_test), shuffle=False)

    net = SNN(input_size=dataset.input_feature_size)
    net.load_state_dict(torch.load("model_data/SNN2_{}.pt".format(filename), map_location=torch.device('cpu'))
                        ['model_state_dict'], strict=False)

    # Give the user the option to load their pretrained weights
    # net.load_state_dict(torch.load("neurobench/examples/primate_reaching/model_data/model_parameters.pth"))

    model = SNNTorchModel(net)

    # metrics = ["r_squared", "model_size", "latency", "MACs"]
    static_metrics = ["model_size"]
    data_metrics = ["r2"]

    # Benchmark expects the following:
    benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, data_metrics])
    results = benchmark.run()
    print(results)
