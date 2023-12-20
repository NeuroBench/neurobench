from neurobench.datasets import WISDMDataLoader
from training import SpikingNetwork
from neurobench.accumulators.accumulator import choose_max_count
from neurobench.benchmarks import Benchmark
from neurobench.models import SNNTorchModel


if __name__ == '__main__':
    breakpoint()
    batch_size = 256
    lr = 1.e-3
    dataset_path = "./data/nehar/watch_subset2_40.npz"
    data_module = WISDMDataLoader(path=dataset_path, batch_size=batch_size)
    data_module.setup('test')

    num_inputs = data_module.num_inputs
    num_outputs = data_module.num_outputs
    num_steps = data_module.num_steps

    spiking_network = SpikingNetwork.load_from_checkpoint('neurobench/examples/nehar/model_data/WISDM_snnTorch.ckpt', map_location='cpu')

    model = SNNTorchModel(spiking_network.model)
    test_set_loader = data_module.test_dataloader()

    # # # postprocessors
    postprocessors = [choose_max_count]
    # #
    static_metrics = ["footprint"]
    workload_metrics = ["classification_accuracy"]
    # #
    benchmark = Benchmark(model, test_set_loader, [], postprocessors, [static_metrics, workload_metrics])
    results = benchmark.run()
    print(results)