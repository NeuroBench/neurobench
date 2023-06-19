# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt
from jitcdde import jitcdde, y, t, jitcdde_lyap


def plot_results(mackeyglass_soln, x_predict_train, x_test, warmup, dt, tau, plottime_tr, plottime_ts, nrmse_train, nrmse_test):
    ##
    ## Visualize data and predictions
    ##
    
    pts_tau = np.floor(tau/dt).astype(int)
    t_linewidth=1.1
    a_linewidth=0.3
    xlabel=np.arange(warmup, warmup+plottime_tr+0.01, 400)
    
    #xlabel_ts=[1000,1200,1400,1600,1800,2000]
    xlabel_ts=np.arange(warmup+traintime, warmup+traintime+plottime_ts+0.01, 400)
    
    plt.rcParams.update({'font.size': 12})
    
    fig1 = plt.figure(dpi=500)
    fig1.set_figheight(8)
    fig1.set_figwidth(14)
    
    h=140
    w=150
    
    # top left of grid is 0,0
    axs1 = plt.subplot2grid(shape=(h,w), loc=(0, 9), colspan=22, rowspan=32) 
    axs2 = plt.subplot2grid(shape=(h,w), loc=(52, 0), colspan=42, rowspan=20)
    
    axs3 = plt.subplot2grid(shape=(h,w), loc=(0, 61), colspan=22, rowspan=32)
    axs4 = plt.subplot2grid(shape=(h,w), loc=(52, 50),colspan=42, rowspan=20)
    
    
    
    # true Mackey-Glass system
    axs1.plot(mackeyglass_soln[0,pts_tau:],mackeyglass_soln[0,0:maxtime_pts-pts_tau],linewidth=a_linewidth)
    axs1.set_xlabel('u(t)', style='italic')
    axs1.set_ylabel('u(t-17)', style='italic')
    axs1.set_title('ground truth')
    axs1.text(-0.76,.92,'a)', ha='left', va='bottom',transform=axs1.transAxes)
    axs1.axes.set_xbound(.3,1.4)
    axs1.axes.set_ybound(.3,1.4)
    
    
    # training phase 
    axs2.set_title('training phase, NRMSE: {:.5f}'.format(nrmse_train)) 
    axs2.plot(t_eval[warmup_pts:warmup_pts+plottime_tr_pts],mackeyglass_soln[0,warmup_pts:warmup_pts+plottime_tr_pts],linewidth=1*t_linewidth, linestyle='solid')
    axs2.plot(t_eval[warmup_pts:warmup_pts+plottime_tr_pts],x_predict_train[0,0:plottime_tr_pts],linewidth=t_linewidth, color='r', linestyle='dashed')
    axs2.set_ylabel('u', style='italic')
    axs2.text(-.155*1.2,0.87,'b)', ha='left', va='bottom',transform=axs2.transAxes)
    #axs2.axes.xaxis.set_ticklabels([])
    axs2.axes.set_ybound(.3,1.4)
    axs2.axes.set_xbound(warmup,warmup+plottime_tr)
    axs2.set_xticks(xlabel)
    axs2.set_xlabel('Time')
    
    
    
    # predicted data
    axs3.plot(x_test[0,1+pts_tau:],x_test[0,1:testtime_pts+1-pts_tau],linewidth=a_linewidth,color='r')
    axs3.set_xlabel('u(t)', style='italic')
    axs3.set_ylabel('u(t-17)', style='italic')
    axs3.set_title('Forecast')
    axs3.text(-0.82,0.92,'c)', ha='left', va='bottom',transform=axs3.transAxes)
    axs3.axes.set_xbound(.3,1.4)
    axs3.axes.set_ybound(.3,1.4)
    
    # testing phase x
    axs4.set_title('testing phase, NRMSE: {:.5f}'.format(nrmse_test))
    #axs4.set_xticks(xlabel)
    axs4.plot(t_eval[warmtrain_pts-1:warmtrain_pts+plottime_pts-1],mackeyglass_soln[0,warmtrain_pts-1:warmtrain_pts+plottime_pts-1],linewidth=t_linewidth)
    axs4.plot(t_eval[warmtrain_pts-1:warmtrain_pts+plottime_pts-1],x_test[0,0:plottime_pts],linewidth=t_linewidth,color='r')
    
    #axs4.set_ylabel('u', style='italic')
    axs4.text(-.155*1.15,0.87,'d)', ha='left', va='bottom',transform=axs4.transAxes)
    axs4.axes.set_xbound(warmup+traintime,warmup+traintime+plottime_ts)
    #axs4.axes.xaxis.set_ticklabels([])
    axs4.axes.set_ybound(.3,1.4)
    #axs4.axes.set_ybound(-10.3,10.4)
    axs4.set_xlabel('Time')
    axs4.set_xticks(xlabel_ts)
        

    
    #plt.savefig('predict_mackey_glass_seed.png')
    plt.show() 


##
## Parameters for generating and visualizing data
##

# Time step for sampling data 
dt = 1.0

# Units of time for pre-training warm-up period
warmup=1000.
# Units of time to train for
traintime=7000 #11000.
# units of time to test for
testtime=2000.

# total time to run for
maxtime = warmup+traintime+testtime


# Lyapunov time of Mackey-Glass system
lyaptime=185 # From numerical estimates 
num_lyaptime = 3.


# discrete-time versions of the times defined above
warmup_pts=round(warmup/dt)
traintime_pts=round(traintime/dt)
warmtrain_pts=warmup_pts+traintime_pts
testtime_pts=round(testtime/dt)
lyaptime_pts=round(num_lyaptime*lyaptime/dt)

t_eval = np.arange(0, maxtime, dt)
maxtime_pts=len(t_eval)

# how much of train or test time to plot
plottime_tr = 2000
plottime_tr_pts=round(plottime_tr/dt)

plottime_ts = 2000
plottime_pts=round(plottime_ts/dt)



##
## Mackey-Glass settings
##
tau = 17
nmg = 10
beta = 0.2
gamma = 0.1
constant_past = 0.9
mackeyglass = [ beta * y(0,t-tau) / (1 + y(0,t-tau)**nmg) - gamma*y(0) ]

# Create the equation object based on the settings
DDE = jitcdde(mackeyglass)
DDE.constant_past([constant_past])
DDE.step_on_discontinuities()



np.random.seed(seed=0)

##
## Generate data from the Mackey-Glass system
##
mackeyglass_soln = np.zeros((1,maxtime_pts))
count = 0
for time in np.arange(DDE.t, DDE.t+maxtime, dt):
    mackeyglass_soln[0,count] = DDE.integrate(time)[0]
    count += 1

# total variance of the Mackey-Glass system
total_var=np.var(mackeyglass_soln[0,:])


##
## Hyperparameters for ESN
##

# Number of input units. Defined by the task
d = 1 

# Number of hidden units in the reservoir
N = 200

# Input weight scale
sigma_in = 1.0

# Input weight scale for the bias
sigma_in_bias = 0.2

# Connection probability
connect_prob = 0.15

# Spectral radius
rho = 1.25

# Leaking rate parameter
leakage = 0.3

# Regularization parameter for ridge regression
ridge_param = 1.e-8



##
## Initialize ESN
##

#Gaussian random projection matrix for feeding input into the reservoir
Win = (np.random.uniform(size=(N,2))-0.5)
Win[:,0] = sigma_in_bias * Win[:,0]
Win[:,1] = sigma_in * Win[:,1]

#Recurrent connectivity matrix W for the reservoir
#Choose nonzero connections 
W = (np.random.uniform(size=(N,N)) <= connect_prob).astype(float)
#Assign random values to these connections
W[W == True] = np.random.normal(size=int(np.sum(W)))
# Scale the resuling recurrent connectivity matrix 
w, _ = np.linalg.eig(W)
W = rho*W/abs(w[0])
# Alternative way to form the recurent connectivity matrix
#W, _ = np.linalg.qr(np.random.normal(size=(N,N)))
#W = rho*W



##
## Training with ESN
##

# Form reservoir states for the warm-up and training data
# Pick the warm-up and training data
x_train = mackeyglass_soln[:,0:warmtrain_pts+1]
# Add constant bias
x_train = np.concatenate((1*np.ones((1, warmtrain_pts+1)), x_train, ), axis=0)


# Obtain the reservoir states for the training phase
reservoir_tr = np.zeros((N+2, warmtrain_pts))
for i in range(1,warmtrain_pts):    
    # Bias & the current input
    reservoir_tr[0:2,i] = x_train[:,i]
    # Update the reservoir
    reservoir_tr[2:,i:i+1] = (1-leakage)*reservoir_tr[2:,i-1:i] + leakage*np.tanh(Win@x_train[:,i:i+1] + W @ reservoir_tr[2:,i-1:i]) #+ 1e-10 * np.random.normal(size=(N,1)) )


# Ridge regression: train the readout matrix Wout to map reservoir_tr to x_train[1,t+1]. Warmup period is ignored
#Wout = (x_train[1:2,warmup_pts+1:warmtrain_pts+1]) @ reservoir_tr[:,warmup_pts:].T @ np.linalg.pinv(reservoir_tr[:,warmup_pts:] @ reservoir_tr[:,warmup_pts:].T + ridge_param*np.identity(N+2))
Wout = np.linalg.lstsq((reservoir_tr[:,warmup_pts:] @ reservoir_tr[:,warmup_pts:].T + ridge_param*np.identity(N+2)), (reservoir_tr[:,warmup_pts:] @ x_train[1:2,warmup_pts+1:warmtrain_pts+1].T), rcond=None)[0].T


##
## Test with ESN
##
  
# Create a place to store predictions
x_test = 1*np.ones((2,testtime_pts+1)) 
# Add constant bias
x_test[:,0] = x_train[:,-2] 
# Initialize the reservoir states
reservoir_ts = np.zeros((N+2, testtime_pts+1))
reservoir_ts[:,0] = reservoir_tr[:,-1]
# Form reservoir states for the test data
for i in range(0,testtime_pts):
    # The first step is skipped as it inhereted from the training data
    if i!=0:
        # Bias & the current input
        reservoir_ts[0:2,i] = x_test[:,i]
        # Update the reservoir
        reservoir_ts[2:,i:i+1] = (1-leakage)*reservoir_ts[2:,i-1:i] + leakage*np.tanh(Win@x_test[:,i:i+1] + W @ reservoir_ts[2:,i-1:i])
    # Do forecast      
    x_test[1,i+1] =   Wout @ reservoir_ts[:,i]
        
      
##
## Compute errors & visualize 
##

# apply Wout to the training reservoir states to get the training predictions
x_predict_train =  Wout @ reservoir_tr[:,warmup_pts:]

# calculate NRMSE between true Mackey-Glass and training output
nrmse_train = np.sqrt(np.mean((mackeyglass_soln[0:1,warmup_pts+1:warmtrain_pts+1]-x_predict_train[:,:])**2)/total_var)

# calculate NRMSE between true Mackey-Glass and prediction for the predefined number of Lyapunov times
nrmse_test = np.sqrt(np.mean((mackeyglass_soln[0:1,warmtrain_pts:warmtrain_pts+lyaptime_pts]-x_test[1:2,1:lyaptime_pts+1])**2)/total_var)

print('training NRMSE ESN: '+str(nrmse_train))
print('testing NRMSE ESN: ' + str(nrmse_test))

plot_results(mackeyglass_soln, x_predict_train, x_test[1:2,:], warmup, dt, tau, plottime_tr, plottime_ts, nrmse_train, nrmse_test)










