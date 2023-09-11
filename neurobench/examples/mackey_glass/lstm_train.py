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
import wandb
import argparse
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

#from neurobench.models.echo_state_network import EchoStateNetwork
from neurobench.models.LSTM_model import LSTMModel
from neurobench.datasets.mackey_glass import MackeyGlass


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--wb', dest='wandb_state', type=str, default="offline", help="wandb state")
    parser.add_argument('--name', type=str, default='LSTM_MG', help='wandb run name')
    parser.add_argument('--project', type=str, default='Neurobench', help='wandb project name')
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=50)
    parser.add_argument('--params_idx', type=int, default=0)
    parser.add_argument('--dropout_rate', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--sw', dest='run_sweep', type=bool, default=False, help="Activate wb sweep run")

    args, unparsed = parser.parse_known_args()

    if args.run_sweep:
        wandb.init(name=args.name,
                   mode=args.wandb_state,
                   config=wandb.config)

        config_wb = wandb.config
    else:
        wandb.init(project=args.project,
                   name=args.name,
                   mode=args.wandb_state)

    #MG_parameters = [(17, 0.9)]
    MG_parameters = [[(17,0.9), (25, 0.3303),(26, 0.8540),(20, 0.8781),(18, 0.4360),(30, 0.5388),(21, 0.4173),(19, 0.6225)][args.params_idx]]

    repeat = 1

    ##
    ## Parameters for generating and visualizing data
    ##

    # Time step for sampling data 
    dt = 1.0
    n_epochs = 200
    # Lyapunov time of Mackey-Glass system. IMPORTANT: NOT FINALIZED
    lyaptime = 185 # From numerical estimates 
    num_lyaptime = 3.
    # discrete-time versions of the times defined above
    lyaptime_pts = round(num_lyaptime*lyaptime/dt)
    plot_fit = False

    # LSTM parameters
    params = {}
    params['input_dim'] = 1
    params['hidden_size'] = args.hidden_size
    params['n_layers'] = args.n_layers
    params['output_dim'] = 1
    params['dropout_rate'] = args.dropout_rate
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
            opt = torch.optim.Adam(lstm.parameters(), lr=args.lr)

            # training loop
            for epoch in range(n_epochs):

                pre = lstm(mackeyglass.training_data)
                loss_val = criterion(pre[mackeyglass.warmup_pts:,:], 
                                         mackeyglass.training_targets)

                opt.zero_grad()
                loss_val.backward()
                opt.step()

                print(f"Epoch {epoch}: loss = {loss_val.item()}")
                wandb.log({"loss": loss_val})
     
            # calculate NRMSE between true Mackey-Glass and train/test prediction for the predefined number of Lyapunov times
            prediction_train = pre[mackeyglass.warmup_pts:,:]
            nrmse_train = torch.sqrt(torch.mean((mackeyglass.training_targets-prediction_train)**2)/mackeyglass.total_var)
                    
            # Update the statistics
            nrmse_train_statistics[i,i_cns] = nrmse_train
                    
            print('Run number ' + str(i) + ':')
            print('training NRMSE LSTM: '+str(nrmse_train.item()))  

            wandb.log({"nrmse_train": nrmse_train.item()})
            torch.save(lstm, 'lstm.pth')

            prediction = torch.zeros((mackeyglass.testtime_pts, lstm.input_dim), dtype=torch.float64)

            # Forecast single_step: model is fed with the true data
            for j in range(0,mackeyglass.testtime_pts):
                sample = mackeyglass.test_data[j:j+1,:]
                sample = lstm(sample)
                prediction[j,:] = sample
            
            #breakpoint()

            # calculate NRMSE between true Mackey-Glass and train/test prediction for the predefined number of Lyapunov times
            nrmse_test = torch.sqrt(torch.mean((mackeyglass.test_data_targets[0:lyaptime_pts,:]-prediction[0:lyaptime_pts,:])**2)/mackeyglass.total_var)
            
            # Update the statistics
            nrmse_test_statistics[i,i_cns] = nrmse_test
            
            print('Run number ' + str(i) + ':')       
            print('testing NRMSE ESN: ' + str(nrmse_test.item()))
            
            wandb.log({"nrmse_test": nrmse_test.item()})

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


if __name__ == '__main__':
    main() 
