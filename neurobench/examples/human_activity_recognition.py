from neurobench.datasets.WISDM_data_loader import WISDMDataModule
from model_data.ConvSNN_HAR import SpikingNetwork
from neurobench.models import SNNTorchModel
from neurobench.accumulators.accumulator import aggregate,choose_max_count
from neurobench.benchmarks import Benchmark


if __name__ == '__main__':
    batch_size = 128
    lr = 1.e-3
    dataset_path = "download your dataset from https://github.com/neuromorphic-polito/NeHAR/blob/main/data/data_watch_subset2_40.npz and store it into a folder"
    data_module = WISDMDataModule(dataset_path, batch_size=batch_size)
    data_module.setup('test')

    num_inputs = data_module.num_inputs
    num_outputs = data_module.num_outputs
    num_steps = data_module.num_steps

    spiking_network = SpikingNetwork.load_from_checkpoint('./model_data/WISDM_snnTorch.ckpt', map_location='cpu')

    model = SNNTorchModel(spiking_network.model)
    test_set_loader = data_module.test_dataloader()

    # # # postprocessors
    postprocessors = [choose_max_count]
    # #
    static_metrics = ["model_size", ]
    data_metrics = ["classification_accuracy",  "multiply_accumulates"]
    # #
    benchmark = Benchmark(model, test_set_loader, [], postprocessors, [static_metrics, data_metrics])
    results = benchmark.run()
    print(results)