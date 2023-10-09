import torch

from torch.utils.data import Subset, DataLoader

import pandas as pd

from neurobench.datasets import MackeyGlass
from neurobench.models import TorchModel
from neurobench.benchmarks import Benchmark

from neurobench.examples.mackey_glass.echo_state_network import EchoStateNetwork

mg_parameters_file="neurobench/datasets/mackey_glass_parameters.csv"
mg_parameters = pd.read_csv(mg_parameters_file)

# generate results for only tau=17
single_series = True
if single_series:
    mg_parameters = mg_parameters[mg_parameters.tau == 17]

# Load hyperparameters of echo state networks found via the random search
esn_parameters = pd.read_csv("neurobench/examples/mackey_glass/echo_state_network_hyperparameters.csv")

# benchmark run over 14 different series
sMAPE_scores = []
synops = []

# Number of simulations to run for each time series
repeat = 30
# Shift time series by 0.5 of its Lyapunov times for each independent run 
start_offset_range = torch.arange(0., 0.5*repeat, 0.5) 

for repeat_id in range(repeat):
    for series_id in range(len(mg_parameters)):
        tau = mg_parameters.tau[series_id]
        mg = MackeyGlass(tau = tau, 
                         lyaptime = mg_parameters.lyapunov_time[series_id],
                         constant_past = mg_parameters.initial_condition[series_id],
                         start_offset=start_offset_range[repeat_id].item(),
                         bin_window=1)
    
        train_set = Subset(mg, mg.ind_train)
        test_set = Subset(mg, mg.ind_test)
        
        #Index of the hyperparamters for the current time-series
        ind_tau = esn_parameters.index[esn_parameters['tau'] == tau].tolist()[0]
    
        ## Fitting Model ##
        seed_id = repeat_id     
        
        # generate results for only tau=17
        if single_series:
            esn = EchoStateNetwork(in_channels=1, 
                reservoir_size = 200, 
                input_scale = torch.tensor([0.2,1], dtype=torch.float64),
                connect_prob = 0.15, 
                spectral_radius = 1.25,
                leakage = 0.3, 
                ridge_param = 1.e-8,
                seed_id = seed_id )

        else:
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
        # torch.save(esn, 'neurobench/examples/mackey_glass/model_data/esn.pth')
         
        ## Load Model ##
        # net = torch.load('neurobench/examples/mackey_glass/model_data/esn.pth')
        test_set_loader = DataLoader(test_set, batch_size=mg.testtime_pts, shuffle=False)
    
        model = TorchModel(esn)
    
        static_metrics = ["model_size", "connection_sparsity"]
        data_metrics = ["sMAPE", "activation_sparsity","synaptic_operations"]
    
        benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, data_metrics]) 
        results = benchmark.run()
        print(results)
        sMAPE_scores.append(results["sMAPE"])
        synops.append(results["synaptic_operations"]["MACs"])

print("Average sMAPE score accross all repeats and time series: ", sum(sMAPE_scores)/len(sMAPE_scores))
print("Average synop MACs accross all repeats and time series: ", sum(synops)/len(synops))

# Scores for tau=17:
# Footprint, Connection Sparsity, Activation Sparsity, SynOps MACs, sMAPE

# ESN best hyperparams: (966,0.9,0.78,0.97,1.19,0.69,1e-07)
# 7488448, 0.03, 0, 908095.2, 0, 9.917

# ESN prev hyperparams (200,0.2,1,0.15,1.25,0.3,1.e-8)
# 324816, 0.8404, 0, 6612.7, 0, 18.634
