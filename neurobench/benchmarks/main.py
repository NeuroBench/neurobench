"""
=====================================================================
Project:      NeuroBench
File:         main.py
Description:  Example Code for how to run Benchmark
Date:         12. May 2023
=====================================================================
Copyright stuff
=====================================================================
"""

from neurobench.datasets.primate_reaching import PrimateReaching
from neurobench.models.SNN_baseline_ETH import SNN
from neurobench.benchmarks.benchmark import Benchmark
import yaml
import torch


if __name__ == '__main__':
    with open('hyperparams.yaml') as f:
        hyperparams = yaml.load(f, Loader=yaml.loader.SafeLoader)

    if torch.cuda.is_available():
        print("using Cuda")
        hyperparams["device"] = torch.device("cuda")
    else:
        print("using CPU")
        hyperparams["device"] = torch.device("cpu")

    ds = PrimateReaching(hyperparams['dataset_file'], biological_delay=140, window=hyperparams['steps'],
                         stride=hyperparams['stride'],
                         splits=hyperparams['splits'])

    net = SNN(192, 2, hyperparams=hyperparams)

    snn_benchmark = Benchmark(dataset=ds, net=net, hyperparams=hyperparams)

    snn_benchmark.run()

    snn_benchmark.result.visualize_learning_curve()
