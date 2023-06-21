"""
=====================================================================
Project:      NeuroBench
File:         example_snn.py
Description:  Example Code for how to run Benchmark
Date:         12. May 2023
=====================================================================
Copyright stuff
=====================================================================
"""

from neurobench.datasets.primate_reaching import PrimateReaching
from neurobench.models.ANN_Arindam import ANNModel
from neurobench.benchmarks.benchmark import Benchmark
import yaml
import torch


if __name__ == '__main__':
    with open('hyperparams_ann.yaml') as f:
        hyperparams = yaml.load(f, Loader=yaml.loader.SafeLoader)

    if torch.cuda.is_available():
        print("using Cuda")
        hyperparams["device"] = torch.device("cuda")
    else:
        print("using CPU")
        hyperparams["device"] = torch.device("cpu")

    ds = PrimateReaching(hyperparams['dataset_file'], window=hyperparams['window'],
                         stride=hyperparams['stride'],
                         splits=hyperparams['splits'], binned=True)

    net = ANNModel(input_dim=192, layer1=32, layer2=48, output_dim=2, dropout_rate=hyperparams['dropout'],
                   hyperparams=hyperparams)

    ann_benchmark = Benchmark(dataset=ds, net=net, hyperparams=hyperparams)

    ann_benchmark.run()

    ann_benchmark.result.visualize_learning_curve()
