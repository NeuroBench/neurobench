import torch

from torch.utils.data import Subset, DataLoader

import pandas as pd

from neurobench.datasets import MackeyGlass
from neurobench.models import TorchModel
from neurobench.benchmarks import Benchmark

from neurobench.examples.mackey_glass.echo_state_network import EchoStateNetwork

mg_parameters_file="neurobench/datasets/mackey_glass_parameters.csv"
mg_parameters = pd.read_csv(mg_parameters_file)


# benchmark run over 14 different series
sMAPE_scores = []

for series_id in range(len(mg_parameters)):
    mg = MackeyGlass(tau = mg_parameters.tau[series_id], 
                     lyaptime = mg_parameters.lyapunov_time[series_id],
                     constant_past = mg_parameters.initial_condition[series_id],
                     start_offset=0.)

    train_set = Subset(mg, mg.ind_train)
    test_set = Subset(mg, mg.ind_test)

    ## Fitting Model ##
    seed_id = 0
    # TODO: refactor the ESN so that it is correct with the static metrics like model_size
    esn = EchoStateNetwork(in_channels=1, 
        reservoir_size = 200, 
        input_scale = torch.tensor([0.2,1],dtype = torch.float64), 
        connect_prob = 0.15, 
        spectral_radius = 1.25, 
        leakage = 0.3, 
        ridge_param = 1.e-8, 
        seed_id = seed_id)
    esn.train()
    train_data, train_labels = train_set[:]
    train_data = train_data.permute(1,0,2) # (batch, timesteps, features)
    warmup = 0.6 # in Lyapunov times
    warmup_pts = round(warmup*mg.pts_per_lyaptime)
    train_labels = train_labels[warmup_pts:]
    esn.fit(train_data, train_labels, warmup_pts)
    torch.save(esn, 'neurobench/examples/model_data/esn.pth')
     
    ## Load Model ##
    net = torch.load('neurobench/examples/model_data/esn.pth')
    test_set_loader = DataLoader(test_set, batch_size=mg.testtime_pts, shuffle=False)

    model = TorchModel(net)

    # data_metrics = ["activation_sparsity", "multiply_accumulates", "sMAPE"]

    static_metrics = ["model_size", "connection_sparsity"]
    data_metrics = ["sMAPE"]

    benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, data_metrics]) 
    results = benchmark.run()
    print(results)
    sMAPE_scores.append(results["sMAPE"])

print("Average sMAPE score: ", sum(sMAPE_scores)/len(sMAPE_scores))

