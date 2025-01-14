import torch

from torch.utils.data import Subset, DataLoader

import pandas as pd

from neurobench.datasets import MackeyGlass
from neurobench.models import TorchModel
from neurobench.benchmarks import Benchmark

from echo_state_network import EchoStateNetwork

from neurobench.metrics.workload import (
    ActivationSparsity,
    SynapticOperations,
    SMAPE
)
from neurobench.metrics.static import (
    Footprint,
    ConnectionSparsity,
)

# Load hyperparameters of echo state networks found via the random search
esn_parameters = pd.read_csv("model_data/echo_state_network_hyperparameters.csv")

# benchmark run over 14 different series
sMAPE_scores = []
synop_macs = []
synop_dense = []

# Number of simulations to run for each time series
repeat = 30
# Shift time series by 0.5 of Lyapunov time-points for each independent run 
start_offset_range = torch.arange(0., 0.5*repeat, 0.5) 
lyaptime_pts = 75
start_offset_range = start_offset_range * lyaptime_pts

# data in repo root dir
data_dir = "../../../data/mackey_glass/"

for tau in range(17, 31):
    for repeat_id in range(repeat):
        filepath = data_dir + "mg_" + str(tau) + ".npy"
        offset = start_offset_range[repeat_id].item()
        mg = MackeyGlass(filepath,
                         start_offset=offset,
                         bin_window=1)
    
        train_set = Subset(mg, mg.ind_train)
        test_set = Subset(mg, mg.ind_test)
        
        #Index of the hyperparamters for the current time-series
        ind_tau = esn_parameters.index[esn_parameters['tau'] == tau].tolist()[0]
    
        ## Fitting Model ##
        seed_id = repeat_id     

        esn = EchoStateNetwork(in_channels=1, 
                reservoir_size = esn_parameters['reservoir_size'][ind_tau], 
                input_scale = torch.tensor([esn_parameters['scale_bias'][ind_tau], esn_parameters['scale_input'][ind_tau],],dtype = torch.float64), 
                connect_prob = esn_parameters['connect_prob'][ind_tau], 
                spectral_radius = esn_parameters['spectral_radius'][ind_tau],
                leakage = esn_parameters['leakage'][ind_tau], 
                ridge_param = esn_parameters['ridge_param'][ind_tau],
                seed_id = seed_id )


        esn.train()
        train_data, train_labels = train_set[:] # outputs (batch, bin_window, 1)
        warmup = 0.6 # in Lyapunov times
        warmup_pts = round(warmup*mg.pts_per_lyaptime)
        train_labels = train_labels[warmup_pts:]
        esn.fit(train_data, train_labels, warmup_pts)
        
        test_set_loader = DataLoader(test_set, batch_size=mg.testtime_pts, shuffle=False)
    
        model = TorchModel(esn)
    
        static_metrics = [Footprint, ConnectionSparsity]
        workload_metrics = [SMAPE, ActivationSparsity, SynapticOperations]
    
        benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, workload_metrics]) 
        results = benchmark.run()
        print(results)
        sMAPE_scores.append(results["SMAPE"])
        synop_macs.append(results["SynapticOperations"]["Effective_MACs"])
        synop_dense.append(results["SynapticOperations"]["Dense"])

print("Average sMAPE score accross all repeats and time series: ", sum(sMAPE_scores)/len(sMAPE_scores))
print("Average synop MACs accross all repeats and time series: ", sum(synop_macs)/len(synop_macs))
print("Average synop dense accross all repeats and time series: ", sum(synop_dense)/len(synop_dense))

# Score for repeat=1
# Average sMAPE score accross all repeats and time series:  40.803375080880265
# Average synop MACs accross all repeats and time series:  290192.71952380956
# Average synop dense accross all repeats and time series:  565148.5714285715
