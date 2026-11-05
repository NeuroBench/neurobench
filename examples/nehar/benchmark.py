import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
from torch.utils.data import DataLoader

from neurobench.datasets import WISDM
from neurobench.benchmarks import Benchmark
from neurobench.models import SNNTorchModel
from neurobench.processors.postprocessors import ChooseMaxCount
from neurobench.metrics.workload import (
    ActivationSparsity,
    ActivationSparsityByLayer,
    MembraneUpdates,
    SynapticOperations,
    ClassificationAccuracy,
    NeuronOperations,
)
from neurobench.metrics.static import (
    ParameterCount,
    Footprint,
    ConnectionSparsity,
)

from SCNN import SCNN


if __name__ == "__main__":

    BATCH_SIZE   = 256
    DATASET_ROOT = os.path.join(SCRIPT_DIR, "../../data/nehar")
    CKPT_PATH    = os.path.join(SCRIPT_DIR, "model_data/WISDM_snnTorch.pt")
    RESULTS_PATH = os.path.join(SCRIPT_DIR, "results")
    ONNX_PATH    = os.path.join(SCRIPT_DIR, "model_data/nehar_snnTorch.onnx")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = WISDM(root=DATASET_ROOT, split="test", download=True)
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=min(8, os.cpu_count() or 1),
        drop_last=False,
        persistent_workers=True,
    )

    scnn = SCNN().to(device)
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    scnn.load_state_dict(checkpoint["model_state_dict"])
    scnn.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_acc={checkpoint['val_acc']:.4f})")

    model = SNNTorchModel(scnn, custom_forward=True)

    dummy_input = torch.randn(1, test_set.n_timesteps, test_set.n_channels)

    postprocessors  = [ChooseMaxCount()]
    static_metrics  = [ParameterCount, Footprint, ConnectionSparsity]
    workload_metrics = [
        ActivationSparsity,
        ActivationSparsityByLayer,
        MembraneUpdates,
        SynapticOperations,
        ClassificationAccuracy,
        NeuronOperations,
    ]

    benchmark = Benchmark(
        model, test_loader, [], postprocessors, [static_metrics, workload_metrics]
    )
    results = benchmark.run(verbose=False)
    print(results)

    benchmark.save_benchmark_results(RESULTS_PATH)
    benchmark.to_onnx(dummy_input, ONNX_PATH)