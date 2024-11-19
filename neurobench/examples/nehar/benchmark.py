from neurobench.datasets import WISDM
from training import SpikingNetwork
from neurobench.postprocessing.postprocessor import choose_max_count
from neurobench.benchmarks import Benchmark
from neurobench.models import SNNTorchModel
from neurobench.benchmarks.metrics.base import StaticMetric
from neurobench.benchmarks.metrics.workload import ActivationSparsity
from neurobench.benchmarks.metrics.workload import MembraneUpdates


if __name__ == "__main__":
    batch_size = 256
    lr = 1.0e-3
    # data in repo root dir
    dataset_path = "../../../data/nehar/watch_subset2_40.npz"
    data_module = WISDM(path=dataset_path, batch_size=batch_size)
    data_module.setup("test")

    num_inputs = data_module.num_inputs
    num_outputs = data_module.num_outputs
    num_steps = data_module.num_steps

    spiking_network = SpikingNetwork.load_from_checkpoint(
        "./model_data/WISDM_snnTorch.ckpt", map_location="cpu"
    )

    model = SNNTorchModel(spiking_network.model)
    test_set_loader = data_module.test_dataloader()

    # # # postprocessors
    postprocessors = [choose_max_count]

    class CustomMetric(StaticMetric):
        def __call__(self, model):
            return sum(p.numel() for p in model.__net__().parameters())

    # #
    static_metrics = ["parameter_count", "footprint", CustomMetric]
    workload_metrics = [ActivationSparsity, MembraneUpdates]
    # #
    benchmark = Benchmark(
        model, test_set_loader, [], postprocessors, [static_metrics, workload_metrics]
    )
    results = benchmark.run(verbose=True)
    print(results)

    results = benchmark.run(verbose=True)
    print(results)
