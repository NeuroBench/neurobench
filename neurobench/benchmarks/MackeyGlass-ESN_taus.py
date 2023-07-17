"""
=====================================================================
Project:      NeuroBench
File:         MackeyGlass-ESN_taus.py
Description:  Python code benchmarking on the Mackey-Glass task
Date:         14. July 2023
=====================================================================
Copyright stuff
=====================================================================
"""


import math
import torch
import matplotlib.pyplot as plt

from neurobench.models.echo_state_network import EchoStateNetwork
from neurobench.datasets.mackey_glass import MackeyGlass



##
## Visualize data and predictions
##
def plot_results(mackeyglass, prediction_train, prediction, nrmse_train, nrmse_test, plottime_tr, plottime_ts, ):
    
    # How many time steps of training or test data to plot
    plottime_tr_pts=round(plottime_tr/mackeyglass.dt)
    plottime_pts=round(plottime_ts/mackeyglass.dt)    
    pts_tau = math.floor(mackeyglass.tau/mackeyglass.dt)
    
    t_linewidth=1.1
    a_linewidth=0.3
    xlabel=torch.arange(mackeyglass.offset*mackeyglass.dt+mackeyglass.warmup, mackeyglass.offset*mackeyglass.dt+mackeyglass.warmup+plottime_tr+0.01, 400)
    
    xlabel_ts=torch.arange(mackeyglass.offset*mackeyglass.dt+mackeyglass.warmup+mackeyglass.traintime, mackeyglass.offset*mackeyglass.dt+mackeyglass.warmup+mackeyglass.traintime+plottime_ts+0.01, 400)
    
    plt.rcParams.update({'font.size': 12})
    
    fig1 = plt.figure(dpi=500)
    fig1.set_figheight(9.2)
    fig1.set_figwidth(16)
    
    h=140
    w=150
    
    # top left of grid is 0,0
    axs1 = plt.subplot2grid(shape=(h,w), loc=(0, 9), colspan=22, rowspan=32) 
    axs2 = plt.subplot2grid(shape=(h,w), loc=(52, 0), colspan=42, rowspan=20)
    
    axs3 = plt.subplot2grid(shape=(h,w), loc=(0, 61), colspan=22, rowspan=32)
    axs4 = plt.subplot2grid(shape=(h,w), loc=(52, 50),colspan=42, rowspan=20)
    
    
    
    # true Mackey-Glass attractor
    axs1.plot(mackeyglass.mackeyglass_soln[mackeyglass.offset+pts_tau:mackeyglass.offset+mackeyglass.singleruntime_pts,0],mackeyglass.mackeyglass_soln[mackeyglass.offset:mackeyglass.offset+mackeyglass.singleruntime_pts-pts_tau,0],linewidth=a_linewidth)
    axs1.set_xlabel('x(t)', style='italic')
    axs1.set_ylabel('x(t-' + str(mackeyglass.tau) + ')', style='italic')
    axs1.set_title('Ground truth')
    axs1.text(-0.76,.92,'a)', ha='left', va='bottom',transform=axs1.transAxes)
    axs1.axes.set_xbound(0.2, 1.5)
    axs1.axes.set_ybound(0.2, 1.5)
    
    
    # training phase 
    axs2.set_title('Training phase, NRMSE: {:.4f}'.format(nrmse_train)) 
    axs2.plot(mackeyglass.t_eval[mackeyglass.offset+mackeyglass.warmup_pts:mackeyglass.offset+mackeyglass.warmup_pts+plottime_tr_pts],mackeyglass.mackeyglass_soln[mackeyglass.offset+mackeyglass.warmup_pts:mackeyglass.offset+mackeyglass.warmup_pts+plottime_tr_pts,0],linewidth=1*t_linewidth, linestyle='solid')
    axs2.plot(mackeyglass.t_eval[mackeyglass.offset+mackeyglass.warmup_pts:mackeyglass.offset+mackeyglass.warmup_pts+plottime_tr_pts],prediction_train[0:plottime_tr_pts,0],linewidth=t_linewidth, color='r', linestyle='dashed')
    axs2.set_ylabel('x', style='italic')
    axs2.text(-.155*1.2,0.87,'b)', ha='left', va='bottom',transform=axs2.transAxes)
    axs2.axes.set_ybound(.3,1.4)
    axs2.axes.set_xbound(mackeyglass.offset*mackeyglass.dt+mackeyglass.warmup,mackeyglass.offset*mackeyglass.dt+mackeyglass.warmup+plottime_tr)
    axs2.set_xticks(xlabel)
    axs2.set_xlabel('Time')
    
    
    
    # predicted attractor
    axs3.plot(prediction[pts_tau:,0],prediction[0:mackeyglass.testtime_pts-pts_tau,0],linewidth=a_linewidth,color='r')
    axs3.set_xlabel('x(t)', style='italic')
    axs3.set_ylabel('x(t-' + str(mackeyglass.tau) + ')', style='italic')
    axs3.set_title('Forecast')
    axs3.text(-0.82,0.92,'c)', ha='left', va='bottom',transform=axs3.transAxes)
    axs3.axes.set_xbound(.3,1.4)
    axs3.axes.set_ybound(0.2, 1.5)
    
    # forecasting phase
    axs4.set_title('Forecasting phase, NRMSE: {:.5f}'.format(nrmse_test))
    axs4.plot(mackeyglass.t_eval[mackeyglass.offset+mackeyglass.warmtrain_pts:mackeyglass.offset+mackeyglass.warmtrain_pts+plottime_pts],mackeyglass.mackeyglass_soln[mackeyglass.offset+mackeyglass.warmtrain_pts:mackeyglass.offset+mackeyglass.warmtrain_pts+plottime_pts,0],linewidth=t_linewidth)
    axs4.plot(mackeyglass.t_eval[mackeyglass.offset+mackeyglass.warmtrain_pts:mackeyglass.offset+mackeyglass.warmtrain_pts+plottime_pts],prediction[0:plottime_pts,0],linewidth=t_linewidth,color='r')    
    axs4.text(-.155*1.15,0.87,'d)', ha='left', va='bottom',transform=axs4.transAxes)
    axs4.axes.set_xbound(mackeyglass.offset*mackeyglass.dt+mackeyglass.warmup+mackeyglass.traintime,mackeyglass.offset*mackeyglass.dt+mackeyglass.warmup+mackeyglass.traintime+plottime_ts)
    axs4.axes.set_ybound(0.2, 1.5)
    axs4.set_xlabel('Time')
    axs4.set_xticks(xlabel_ts)
            
    plt.show() 
##
## END Visualize data and predictions
##


# Parameters for several Mackey-Glass time-series (tau,initial condition)
MG_parameters = [(25, 0.3303),(26, 0.8540),(20, 0.8781),(18, 0.4360),(30, 0.5388),(17, 0.9),(21, 0.4173),(19, 0.6225),]

# Number of simulations to run for each time-series
repeat = 5


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
        
        # ESN training phase
        esn.fit(mackeyglass.training_data, mackeyglass.training_targets, mackeyglass.warmup_pts)
    
        # Forecasting phase
        # Create a placeholder to store predictions
        prediction = torch.zeros((mackeyglass.testtime_pts,esn.in_channels), dtype=torch.float64)
        
        #  Forecast with ESN in the autonomous mode
        sample = mackeyglass.training_data[-1:,:]
        for j in range(0,mackeyglass.testtime_pts):
            sample = esn(sample)
            prediction[j,:] = sample
            #sample = mackeyglass.test_data[j:j+1,:]
        
        # calculate NRMSE between true Mackey-Glass and train/test prediction for the predefined number of Lyapunov times
        nrmse_train = torch.sqrt(torch.mean((mackeyglass.training_targets-esn.prediction_train)**2)/mackeyglass.total_var)
        nrmse_test = torch.sqrt(torch.mean((mackeyglass.test_data[0:lyaptime_pts,:]-prediction[0:lyaptime_pts,:])**2)/mackeyglass.total_var)
        
        # Update the statistics
        nrmse_train_statistics[i,i_cns] = nrmse_train
        nrmse_test_statistics[i,i_cns] = nrmse_test
        
        print('Run number ' + str(i) + ':')
        print('training NRMSE ESN: '+str(nrmse_train.item()))        
        print('testing NRMSE ESN: ' + str(nrmse_test.item()))
        
        # Visualize data and predictions
        plot_results(mackeyglass, esn.prediction_train, prediction, nrmse_train, nrmse_test, 2000, 2000,)
    
    
    print('Median results:')
    print('Median training NRMSE ESN: '+str(torch.median(nrmse_train_statistics, dim = 0).values))
    print('Median testing NRMSE ESN: ' + str(torch.median(nrmse_test_statistics, dim = 0).values))
    
    
