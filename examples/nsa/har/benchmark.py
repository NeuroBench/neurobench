"""
Benchmark the NSA HAR LIF baseline with NeuroBench.

Loads a checkpoint from ``train.py`` (or a raw NSA checkpoint) and runs NeuroBench's
static and workload metrics on NSA's HAR test split.

Usage:
    python benchmark.py
    python benchmark.py --checkpoint model_data/har_lif_rec.pt
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # examples/nsa, for nsa_common

from neurobench.benchmarks import Benchmark  # noqa: E402
from neurobench.metrics.static import (  # noqa: E402
    ConnectionSparsity,
    Footprint,
    ParameterCount,
)
from neurobench.metrics.workload import (  # noqa: E402
    ActivationSparsity,
    ClassificationAccuracy,
    SynapticOperations,
)

from data import load_split  # noqa: E402
from model import load_model  # noqa: E402
from nsa_common import wrap_for_neurobench  # noqa: E402

DEFAULT_DATA = os.path.join(HERE, "..", "..", "..", "data", "nsa", "har")
DEFAULT_CKPT = os.path.join(HERE, "model_data", "har_lif_ff.pt")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-path", default=DEFAULT_DATA)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_benchmark(net, dataloader):
    """Wrap an NSA HAR network and set up the NeuroBench benchmark."""
    model = wrap_for_neurobench(net, readout="mean")
    postprocessors = [lambda logits: logits.argmax(-1)]
    static_metrics = [ParameterCount, Footprint, ConnectionSparsity]
    workload_metrics = [ClassificationAccuracy, ActivationSparsity, SynapticOperations]
    return Benchmark(
        model, dataloader, [], postprocessors, [static_metrics, workload_metrics]
    )


def main():
    args = parse_args()
    net = load_model(args.checkpoint)
    test_set = load_split(args.data_path, "test")
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    benchmark = build_benchmark(net, loader)
    results = benchmark.run(device=args.device)
    print(json.dumps(results, indent=2))

    results_path = os.path.join(HERE, "results")
    os.makedirs(results_path, exist_ok=True)
    benchmark.save_benchmark_results(os.path.join(results_path, "har_lif"))


if __name__ == "__main__":
    main()
