"""
=====================================================================
Project:      NeuroBench
File:         MackeyGlass-ESN_taus.py
Description:  Python code benchmarking on the Mackey-Glass task
Date:         20. July 2023
=====================================================================
Copyright stuff
=====================================================================
"""

import sys
sys.path.append("../../..")

import math
import torch
import matplotlib.pyplot as plt

from neurobench.models.echo_state_network import EchoStateNetwork
from neurobench.datasets.mackey_glass import MackeyGlass


MG_parameters = [(17, 0.9)]
repeat = 1

##
## Parameters for generating and visualizing data
##


# Time step for sampling data 
dt = 1.0
# Lyapunov time of Mackey-Glass system. IMPORTANT: NOT FINALIZED
lyaptime=185 # From numerical estimates 
num_lyaptime = 3.
# discrete-time versions of the times defined above
lyaptime_pts=round(num_lyaptime*lyaptime/dt)


# Collect the obtained statistics of models performance
nrmse_train_statistics = torch.zeros((repeat,len(MG_parameters)), dtype=torch.float64)
nrmse_test_statistics =  torch.zeros((repeat,len(MG_parameters)), dtype=torch.float64)


# Loop over all time-series in MG_parameters
for i_cns in range(len(MG_parameters)):
    print(i_cns)
    
    # Generate the data
    tau = MG_parameters[i_cns][0]
    constant_past = MG_parameters[i_cns][1]
    mackeyglass = MackeyGlass(repeat, tau, constant_past, dt = dt)
    
    # Loop over all repeats for the current time-series
    for i in range(0,repeat):
        # Split data for the current repeat into training and test data
        mackeyglass.split_data(i)
        
        # Set seed for initializing the model
        seed_id = i
        
        # Initialize an ESN
        esn = EchoStateNetwork(in_channels=1, reservoir_size = 200, input_scale = torch.tensor([0.2,1],dtype = torch.float64), connect_prob = 0.15, spectral_radius = 1.25, leakage = 0.3, ridge_param = 1.e-8, seed_id = seed_id )
        breakpoint()
        # ESN training phase
        esn.train()

        esn.fit(mackeyglass.training_data, mackeyglass.training_targets, mackeyglass.warmup_pts)
        
        # calculate NRMSE between true Mackey-Glass and train/test prediction for the predefined number of Lyapunov times
        nrmse_train = torch.sqrt(torch.mean((mackeyglass.training_targets-esn.prediction_train)**2)/mackeyglass.total_var)
                
        # Update the statistics
        nrmse_train_statistics[i,i_cns] = nrmse_train
                
        print('Run number ' + str(i) + ':')
        print('training NRMSE ESN: '+str(nrmse_train.item()))  

        torch.save(esn, 'esn.pth')
            