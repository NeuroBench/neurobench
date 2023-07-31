"""
=====================================================================
Project:      NeuroBench
File:         MackeyGlass-LSTM_taus.py
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
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

#from neurobench.models.echo_state_network import EchoStateNetwork
from neurobench.models.LSTM_model import LSTMModel
from neurobench.datasets.mackey_glass import MackeyGlass


MG_parameters = [(17, 0.9)]
repeat = 1

##
## Parameters for generating and visualizing data
##

# Time step for sampling data 
dt = 1.0
n_epochs = 2#00
# Lyapunov time of Mackey-Glass system. IMPORTANT: NOT FINALIZED
lyaptime = 185 # From numerical estimates 
num_lyaptime = 3.
# discrete-time versions of the times defined above
lyaptime_pts = round(num_lyaptime*lyaptime/dt)
plot_fit = False

# LSTM parameters
params = {}
params['input_dim'] = 1
params['hidden_size'] = 50
params['n_layers'] = 2
params['output_dim'] = 1
params['dtype'] = torch.float64

# Collect the obtained statistics of models performance
nrmse_train_statistics = torch.zeros((repeat, len(MG_parameters)), dtype=torch.float64)
nrmse_test_statistics =  torch.zeros((repeat, len(MG_parameters)), dtype=torch.float64)

# Loop over all time-series in MG_parameters
for i_cns in range(len(MG_parameters)):
    print("Parameters set:", MG_parameters[i_cns])
    
    # Generate the data
    tau = MG_parameters[i_cns][0]
    constant_past = MG_parameters[i_cns][1]
    mackeyglass = MackeyGlass(repeat, tau, constant_past, dt = dt)
    
    # Loop over all repeats for the current time-series
    for i in range(0, repeat):
        # Split data for the current repeat into training and test data
        mackeyglass.split_data(i)
        
        # Set seed for initializing the model
        seed_id = i
        
        # Initialize an LSTM model
        lstm = LSTMModel(**params)
        
        # LSTM training phase
        lstm.train()

        criterion = nn.MSELoss()
        opt = torch.optim.Adam(lstm.parameters(), lr=0.01)

        # training loop
        for epoch in range(n_epochs):

            pre = lstm(mackeyglass.training_data)
            loss_val = criterion(pre[mackeyglass.warmup_pts:,:], 
                                     mackeyglass.training_targets)

            opt.zero_grad()
            loss_val.backward()
            opt.step()

            print(f"Epoch {epoch}: loss = {loss_val.item()}")
 
        # calculate NRMSE between true Mackey-Glass and train/test prediction for the predefined number of Lyapunov times
        prediction_train = pre[mackeyglass.warmup_pts:,:]
        nrmse_train = torch.sqrt(torch.mean((mackeyglass.training_targets-prediction_train)**2)/mackeyglass.total_var)
                
        # Update the statistics
        nrmse_train_statistics[i,i_cns] = nrmse_train
                
        print('Run number ' + str(i) + ':')
        print('training NRMSE LSTM: '+str(nrmse_train.item()))  

        torch.save(lstm, 'lstm.pth')

        # plot fit
        if plot_fit:
            print("Plotting training fit")
            plt.figure()
            
            plt.rcParams['font.size'] = 8 
            plt.rcParams['legend.fontsize'] = 6 

            x = np.arange(mackeyglass.training_data.size()[0])

            plt.plot(x, mackeyglass.training_data.cpu().detach().numpy()[:,0], lw=2, label='input')
            plt.plot(x[mackeyglass.warmup_pts:], prediction_train.cpu().detach().numpy()[:,0], lw=2, label='prediction')
            plt.plot(x[mackeyglass.warmup_pts:], mackeyglass.training_targets.cpu().detach().numpy()[:,0], lw=2, label='target')

            plt.xlabel("time steps")
            plt.ylabel("signal amplitude")

            plt.legend()
            print(f" save figure: lstm_fit_{n_epochs}.png") 
            plt.savefig(f"lstm_fit_{n_epochs}.png", dpi=600) 
