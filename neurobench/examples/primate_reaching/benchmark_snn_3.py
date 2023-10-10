import torch
from torch.utils.data import DataLoader, Subset

from neurobench.datasets import PrimateReaching
from neurobench.models.snntorch_models import SNNTorchModel
from neurobench.benchmarks import Benchmark
from neurobench.examples.primate_reaching.SNN_3 import SNNModel3

# Download data to /data/primate_reaching/PrimateReachingDataset. See PrimateReaching
# class for download instructions.

all_files = ["indy_20160622_01", "indy_20160630_01", "indy_20170131_02", 
             "loco_20170210_03", "loco_20170217_02", "loco_20170301_05"]

for filename in all_files:
    # The dataloader and preprocessor has been combined together into a single class
    dataset = PrimateReaching(file_path="data/primate_reaching/PrimateReachingDataset/", filename=filename,
                            num_steps=1, train_ratio=0.5, bin_width=0.004,
                            biological_delay=0, remove_segments_inactive=True)
    test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=len(dataset.ind_test), shuffle=False)

    net = SNNModel3(input_size=dataset.input_feature_size)

    # Give the user the option to load their pretrained weights
    # TODO: currently model is not trained
    net.load_state_dict(torch.load("neurobench/examples/primate_reaching/model_data/snn3_model_parameters/"+filename+"_model_params.pth"))

    model = SNNTorchModel(net)

    # metrics = ["r_squared", "model_size", "latency", "MACs"]
    static_metrics = ["model_size"]
    data_metrics = ["r2"]

    # Benchmark expects the following:
    benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, data_metrics])
    results = benchmark.run()
    print(results)
