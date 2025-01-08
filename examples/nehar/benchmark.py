from neurobench.datasets import WISDM
from training import SpikingNetwork
from neurobench.processors.postprocessors import ChooseMaxCount
from neurobench.benchmarks import Benchmark
from neurobench.models import SNNTorchModel
import torch
from neurobench.metrics.workload import (
    ActivationSparsity,
    MembraneUpdates,
    SynapticOperations,
    ClassificationAccuracy
)
from neurobench.metrics.static import (
    ParameterCount,
    Footprint,
    ConnectionSparsity,
)
import time


if __name__ == "__main__":
    batch_size = 256
    lr = 1.0e-3
    # data in repo root dir
    dataset_path = "../../data/nehar/watch_subset2_40.npz"
    data_module = WISDM(path=dataset_path, batch_size=batch_size)
    data_module.setup("test")

    num_inputs = data_module.num_inputs
    num_outputs = data_module.num_outputs
    num_steps = data_module.num_steps

    spiking_network = SpikingNetwork.load_from_checkpoint(
        "./model_data/WISDM_snnTorch.ckpt", map_location="cpu"
    )

    model = SNNTorchModel(spiking_network.model, custom_forward=False)
    test_set_loader = data_module.test_dataloader()

    # # # postprocessors
    postprocessors = [ChooseMaxCount()]

    # #
    static_metrics = [ParameterCount, Footprint, ConnectionSparsity]
    workload_metrics = [ActivationSparsity, MembraneUpdates, SynapticOperations, ClassificationAccuracy]
    # #
    benchmark = Benchmark(
        model, test_set_loader, [], postprocessors, [static_metrics, workload_metrics]
    )
    start_time = time.time()
    results = benchmark.run(verbose=False)
    print(results)
    #benchmark.save_benchmark_results("./results", file_format="txt")
    dummy_input = torch.randn(1, num_steps,  num_inputs)


    print(f"Time taken: {time.time() - start_time}")
    #benchmark.to_nir(dummy_input, "model_data/nehar_snnTorch.nir")
    print()


