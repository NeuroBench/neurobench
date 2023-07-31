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

from fvcore.nn import FlopCountAnalysis, flop_count_table

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
        
        # Load ESN
        esn = torch.load('esn.pth')

        # Forecasting phase
        # Create a placeholder to store predictions
        prediction = torch.zeros((mackeyglass.testtime_pts,esn.in_channels), dtype=torch.float64)
        
        esn.eval()

        for param in esn.parameters():
            param.requires_grad = False

        mode = "autonomous"

        # Forecast autonomous: model is fed with its own output
        if mode == "autonomous":
            sample = mackeyglass.test_data[0:1,:]
            for j in range(0,mackeyglass.testtime_pts):
                sample = esn(sample)

                # TODO: for stateful network does the flopcounter forward call impact the network state?
                rand = torch.randn_like(sample)
                flops = FlopCountAnalysis(esn, (rand,))

                breakpoint()
                out = flops.total()

                prediction[j,:] = sample

        # Forecast single_step: model is fed with the true data
        elif mode == "single_step":
            for j in range(0,mackeyglass.testtime_pts):
                sample = mackeyglass.test_data[j:j+1,:]
                sample = esn(sample)
                prediction[j,:] = sample
        
        # breakpoint()

        # calculate NRMSE between true Mackey-Glass and train/test prediction for the predefined number of Lyapunov times
        nrmse_test = torch.sqrt(torch.mean((mackeyglass.test_data_targets[0:lyaptime_pts,:]-prediction[0:lyaptime_pts,:])**2)/mackeyglass.total_var)
        
        # Update the statistics
        nrmse_test_statistics[i,i_cns] = nrmse_test
        
        print('Run number ' + str(i) + ':')       
        print('testing NRMSE ESN: ' + str(nrmse_test.item()))
        

        #### Example interface with Benchmark, metrics ####
        '''
        # this dataset should be form (data, targets), where data is tensor of shape (batch, timesteps, features), targets should be (batch, ...)
        # --> ? for this batch should be 1, so data shape should be (1, num_test_steps, 1), targets shape should be (1, num_test_steps) ?
        test_set = mackeyglass.test_data
        # latency for this task doesn't seem very meaningful
        benchmark = Benchmark(model, test_set, [], ["NRMSE", "model_size", "latency", "MACs"]) 
        results = benchmark.run()
        print(results)
        '''
