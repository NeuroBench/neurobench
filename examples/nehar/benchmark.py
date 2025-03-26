import os
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
    ClassificationAccuracy,
    ActivationSparsityByLayer,
)
from neurobench.metrics.static import (
    ParameterCount,
    Footprint,
    ConnectionSparsity,
)


if __name__ == "__main__":
    batch_size = 256
    lr = 1.0e-3

    file_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(file_path, "model_data/WISDM_snnTorch.ckpt")
    dataset_path = os.path.join(file_path, "../../data/nehar/watch_subset2_40.npz") # data in repo root dir

    data_module = WISDM(path=dataset_path, batch_size=batch_size)
    data_module.setup("test")

    num_inputs = data_module.num_inputs
    num_outputs = data_module.num_outputs
    num_steps = data_module.num_steps

    spiking_network = SpikingNetwork.load_from_checkpoint(
        model_path, map_location="cpu"
    )

    model = SNNTorchModel(spiking_network.model, custom_forward=True)
    test_set_loader = data_module.test_dataloader()

    dummy_input = torch.randn(1, num_steps, num_inputs)

    # # # postprocessors
    postprocessors = [ChooseMaxCount()]

    # #
    static_metrics = [ParameterCount, Footprint, ConnectionSparsity]
    workload_metrics = [ActivationSparsity, ActivationSparsityByLayer,MembraneUpdates, SynapticOperations, ClassificationAccuracy]
    # #
    benchmark = Benchmark(
        model, test_set_loader, [], postprocessors, [static_metrics, workload_metrics]
    )
    results = benchmark.run(verbose=True)

    results_path = os.path.join(file_path, "results")
    benchmark.save_benchmark_results(results_path)

    # nir_path = os.path.join(file_path, "model_data/nehar_snnTorch.nir")
    # benchmark.to_nir(dummy_input, nir_path)

    onnx_path = os.path.join(file_path, "model_data/nehar_snnTorch.onnx")
    benchmark.to_onnx(dummy_input, onnx_path)


