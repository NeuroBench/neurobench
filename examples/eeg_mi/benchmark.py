import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import snntorch as snn
from snntorch import surrogate

from neurobench.benchmarks import Benchmark
from neurobench.models import SNNTorchModel
from neurobench.metrics.static import Footprint, ParameterCount, ConnectionSparsity
from neurobench.metrics.workload import (
    ClassificationAccuracy,
    ActivationSparsity,
    SynapticOperations,
)

from neurobench.processors.postprocessors import ChooseMaxCount

from model import EEG_SNN
from neurobench.datasets import ThorEEGMI

DEVICE = "cpu"


val_set = ThorEEGMI(root="../../data/eeg_mi", split="val", download=True)
loader = DataLoader(val_set, batch_size=64, shuffle=False)

# Load model
net = EEG_SNN().to(DEVICE)
net.load_state_dict(torch.load("./model_data/best_model.pt", map_location=DEVICE))

nb_model = SNNTorchModel(net, custom_forward=True)

benchmark = Benchmark(
    model=nb_model,
    dataloader=loader,
    preprocessors=[],
    postprocessors=[ChooseMaxCount()],
    metric_list=[
        [Footprint, ParameterCount, ConnectionSparsity],
        [ClassificationAccuracy, ActivationSparsity, SynapticOperations],
    ],
)

print("\nBenchmark results:", benchmark.run())