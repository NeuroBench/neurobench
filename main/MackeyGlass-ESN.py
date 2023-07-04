# -*- coding: utf-8 -*-
"""

"""

import math
import numpy as np
import matplotlib.pyplot as plt
from jitcdde import jitcdde, y, t, jitcdde_lyap


class EchoStateNetwork():
    """Class for Echo State Networks that creates a network and includes methods for training the network and for performing the inference with the trained model. For details of the architecture please refer to `A Practical Guide to Applying Echo State Networks <https://link.springer.com/chapter/10.1007/978-3-642-35289-8_36>`_.

    Args:
        in_channels (int): the dimensionality of the input time-series.
        reservoir_size (int): the dimensionality of the reservoir. It does not include additional nodes that would be dedicated in case include_input is set to ``True``.                 
        input_scale (float, optional): weight scale for randomly projecting input into the reservoir. Could also be a vector with one scalar per input channel. Default: ``1.0``.
        connect_prob (float, optional): connection probability wihtin the recurrent connectivity matrix. Default: ``0.1``.
        spectral_radius (float, optional): largest spectral radius of the recurrent connectivity matrix. Default: ``1.0``.
        leakage (float, optional): parameter controlling the leaking rate. Default: ``1.0``. 
        ridge_param (float, optional): parameter used when obtaining the readout matrix with the ridge regression. Default: ``0.0``.          
        include_bias (bool, optional)): Whether to use a dedicated input node streaming a constant to the reservoir. Default: ``True``.
        include_input (bool, optional)): Whether to have separate nodes that will explicitly extend the current reservoir state with the current input. Default: ``True``.    
        seed_id (int, optional): parameter used to initialize the pseudo-random generator. Default: ``0``.          

    """

    def __init__(
        self,
        in_channels: int,
        reservoir_size: int,
        input_scale: float = 1.0,
        connect_prob: float = 0.1,
        spectral_radius: float = 1.0,
        leakage: float = 1.0,
        ridge_param: float = 0.0,
        include_bias: bool = True,
        include_input: bool = True,
        seed_id: int = 0,      
    ):
        self.in_channels = in_channels
        self.reservoir_size = reservoir_size
        self.input_scale = input_scale
        self.connect_prob = connect_prob
        self.spectral_radius = spectral_radius
        self.leakage = leakage
        self.ridge_param = ridge_param
        self.include_bias = include_bias
        self.include_input = include_input
        
        # Check if bias is added into the input
        if self.include_bias:
            self.in_channels_bias = self.in_channels + 1 
        else:
            self.in_channels_bias = self.in_channels
            
        # Check if the reservoir state is extended by the current input
        if self.include_input:
            self.reservoir_extension = self.in_channels_bias 
        else:
            self.reservoir_extension = 0
        
        ##
        ## Initialize ESN parameters based on randomness
        ##        
        
        # Set seed
        np.random.seed(seed=seed_id)
        
        #Uniform random projection matrix for feeding input into the reservoir   
        self.Win = (np.random.uniform(size=(self.reservoir_size,self.in_channels_bias))-0.5) 
        #Scaling Win by input_scale
        self.Win = self.input_scale*self.Win

        #Recurrent connectivity matrix W for the reservoir
        #Choose nonzero connections 
        self.W = (np.random.uniform(size=(self.reservoir_size,self.reservoir_size)) <= self.connect_prob).astype(float)
        #Assign random values to these connections
        self.W[self.W == True] = np.random.normal(size=int(np.sum(self.W)))
        # Scale the resuling recurrent connectivity matrix 
        w, _ = np.linalg.eig(self.W)
        self.W = self.spectral_radius*self.W/abs(w[0])
        # Alternative way to form the orthonormal recurrent connectivity matrix
        #self.W, _ = np.linalg.qr(np.random.normal(size=(self.reservoir_size,self.reservoir_size)))
        #self.W = self.spectral_radius*self.W     


    # Performs the training phase of the Echo State Network.
    def fit(
        self, 
        training_data: np.ndarray, 
        targets: np.ndarray, 
        warmup_pts: int,
    ):
        warmtrain_pts = training_data.shape[1]
        
        # Form reservoir states for the warm-up and training data
        # Pick the warm-up and training data
        # Add constant bias if applicable
        if self.include_bias:
            training_data = np.concatenate((1*np.ones((1, warmtrain_pts)), training_data, ), axis=0)
            
        # Initialize reservoir's state
        self.reservoir = np.zeros((self.reservoir_size+self.reservoir_extension, 1))

        # Obtain the reservoir states for the training phase
        self.reservoir_tr = np.zeros((self.reservoir_size+self.reservoir_extension, warmtrain_pts))
        for i in range(0,warmtrain_pts):    
            # Bias & the current input if applicable
            if self.include_input:
                self.reservoir[0:self.reservoir_extension,0] = training_data[:,i]
            # Update the reservoir
            self.reservoir[self.reservoir_extension:,0:1] = (1-self.leakage)*self.reservoir[self.reservoir_extension:,0:1] + self.leakage*np.tanh(self.Win@training_data[:,i:i+1] + self.W @ self.reservoir[self.reservoir_extension:,0:1])
            
            self.reservoir_tr[:,i:i+1] = self.reservoir
            
        # Ridge regression: train the readout matrix Wout to map reservoir_tr to targets. The warmup period defined by warmup_pts is ignored
        #self.Wout = targets @ self.reservoir_tr[:,warmup_pts:].T @ np.linalg.pinv(self.reservoir_tr[:,warmup_pts:] @ self.reservoir_tr[:,warmup_pts:].T + self.ridge_param*np.identity(self.reservoir_size+self.reservoir_extension))
        self.Wout = np.linalg.lstsq((self.reservoir_tr[:,warmup_pts:] @ self.reservoir_tr[:,warmup_pts:].T + self.ridge_param*np.identity(self.reservoir_size+self.reservoir_extension)), (self.reservoir_tr[:,warmup_pts:] @ targets.T), rcond=None)[0].T
        
        self.prediction_train =  self.Wout @ self.reservoir_tr[:,warmup_pts-1:-1]

    ##
    ## Forecast with ESN for a single step in the autonomous mode
    ##        
    def __call__(self):
        
        # Make predictions based on the current reservoir state
        prediction = self.Wout @ self.reservoir
        
        # Update the reservoir state for the next predicition 
        if self.include_bias:
            sample = np.concatenate((1*np.ones((1, 1)), prediction,), axis=0)
        else:
            sample = prediction
        
        if self.include_input:
            self.reservoir[0:self.reservoir_extension,0:1] = sample[:]            
        
        # Update the reservoir
        self.reservoir[self.reservoir_extension:,0:1] = (1-self.leakage)*self.reservoir[self.reservoir_extension:,0:1] + self.leakage*np.tanh(self.Win@sample[:,0:1] + self.W @ self.reservoir[self.reservoir_extension:,0:1])
    
        return prediction

    
    

##
## Visualize data and predictions
##
def plot_results(t_eval, mackeyglass_soln, prediction_train, prediction, offset, warmup, traintime, dt, tau, plottime_tr, plottime_ts, nrmse_train, nrmse_test):

    
    pts_tau = np.floor(tau/dt).astype(int)
    t_linewidth=1.1
    a_linewidth=0.3
    xlabel=np.arange(offset*dt+warmup, offset*dt+warmup+plottime_tr+0.01, 400)
    
    #xlabel_ts=[1000,1200,1400,1600,1800,2000]
    xlabel_ts=np.arange(offset*dt+warmup+traintime, offset*dt+warmup+traintime+plottime_ts+0.01, 400)
    
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
    axs1.plot(mackeyglass_soln[0,offset+pts_tau:offset+singleruntime_pts],mackeyglass_soln[0,offset:offset+singleruntime_pts-pts_tau],linewidth=a_linewidth)
    axs1.set_xlabel('x(t)', style='italic')
    axs1.set_ylabel('x(t-' + str(tau) + ')', style='italic')
    axs1.set_title('Ground truth')
    axs1.text(-0.76,.92,'a)', ha='left', va='bottom',transform=axs1.transAxes)
    axs1.axes.set_xbound(.3,1.4)
    axs1.axes.set_ybound(.3,1.4)
    
    
    # training phase 
    axs2.set_title('Training phase, NRMSE: {:.4f}'.format(nrmse_train)) 
    axs2.plot(t_eval[offset+warmup_pts:offset+warmup_pts+plottime_tr_pts],mackeyglass_soln[0,offset+warmup_pts:offset+warmup_pts+plottime_tr_pts],linewidth=1*t_linewidth, linestyle='solid')
    axs2.plot(t_eval[offset+warmup_pts:offset+warmup_pts+plottime_tr_pts],prediction_train[0,0:plottime_tr_pts],linewidth=t_linewidth, color='r', linestyle='dashed')
    axs2.set_ylabel('x', style='italic')
    axs2.text(-.155*1.2,0.87,'b)', ha='left', va='bottom',transform=axs2.transAxes)
    axs2.axes.set_ybound(.3,1.4)
    axs2.axes.set_xbound(offset*dt+warmup,offset*dt+warmup+plottime_tr)
    axs2.set_xticks(xlabel)
    axs2.set_xlabel('Time')
    
    
    
    # predicted attractor
    axs3.plot(prediction[0,pts_tau:],prediction[0,0:testtime_pts-pts_tau],linewidth=a_linewidth,color='r')
    axs3.set_xlabel('x(t)', style='italic')
    axs3.set_ylabel('x(t-' + str(tau) + ')', style='italic')
    axs3.set_title('Forecast')
    axs3.text(-0.82,0.92,'c)', ha='left', va='bottom',transform=axs3.transAxes)
    axs3.axes.set_xbound(.3,1.4)
    axs3.axes.set_ybound(.3,1.4)
    
    # forecasting phase
    axs4.set_title('Forecasting phase, NRMSE: {:.5f}'.format(nrmse_test))
    axs4.plot(t_eval[offset+warmtrain_pts:offset+warmtrain_pts+plottime_pts],mackeyglass_soln[0,offset+warmtrain_pts:offset+warmtrain_pts+plottime_pts],linewidth=t_linewidth)
    axs4.plot(t_eval[offset+warmtrain_pts:offset+warmtrain_pts+plottime_pts],prediction[0,0:plottime_pts],linewidth=t_linewidth,color='r')    
    axs4.text(-.155*1.15,0.87,'d)', ha='left', va='bottom',transform=axs4.transAxes)
    axs4.axes.set_xbound(offset*dt+warmup+traintime,offset*dt+warmup+traintime+plottime_ts)
    axs4.axes.set_ybound(.3,1.4)
    axs4.set_xlabel('Time')
    axs4.set_xticks(xlabel_ts)
            
    plt.show() 


# Numer of simulations to run for
repeat = 10 

##
## Parameters for generating and visualizing data
##

# Time step for sampling data 
dt = 1.0

# Units of time for pre-training warm-up period
warmup=1000.
# Units of time to train for
traintime=7000. 
# Units of time to forecast for
testtime=2000.
# Units of time to forecast for
runspantime = 1000.

# total time to run for in a single simulation
singleruntime = warmup+traintime+testtime
# Total time to simulate the system
maxtime = (repeat-1)*runspantime + singleruntime



# Lyapunov time of Mackey-Glass system
lyaptime=185 # From numerical estimates 
num_lyaptime = 3.


# discrete-time versions of the times defined above
warmup_pts=round(warmup/dt)
traintime_pts=round(traintime/dt)
warmtrain_pts=warmup_pts+traintime_pts
testtime_pts=round(testtime/dt)
lyaptime_pts=round(num_lyaptime*lyaptime/dt)
runspantime_pts = math.floor(runspantime/dt)
singleruntime_pts = round(singleruntime/dt)
t_eval = np.arange(0, maxtime, dt)
maxtime_pts=len(t_eval)

# how much of train or test time to plot
plottime_tr = 2000
plottime_tr_pts=round(plottime_tr/dt)

plottime_ts = testtime
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


nrmse_train_statistics = np.zeros(repeat)
nrmse_test_statistics = np.zeros(repeat)


for i in range(0,repeat):
    offset = i*runspantime_pts
    seed_id = i
    
    ##
    ## Initialize ESN
    ##
    esn = EchoStateNetwork(in_channels=1, reservoir_size = 200, input_scale = np.array([0.2,1]), connect_prob = 0.15, spectral_radius = 1.25, leakage = 0.3, ridge_param = 1.e-8, seed_id = seed_id )
    
    # Training phase

    training_data = mackeyglass_soln[:,offset:offset+warmtrain_pts]
    targets = mackeyglass_soln [:,offset+warmup_pts+1:offset+warmtrain_pts+1]
    esn.fit(training_data, targets, warmup_pts)

    # Forecasting phase
    # Create a placeholder to store predictions
    prediction = np.zeros((esn.in_channels, testtime_pts))
    
    #  Forecast with ESN in the autonomous mode
    for j in range(0,testtime_pts):
        prediction[:,j] = esn()
    
    # calculate NRMSE between true Mackey-Glass and train/test prediction for the predefined number of Lyapunov times
    nrmse_train = np.sqrt(np.mean((mackeyglass_soln[0:1,offset+warmup_pts:offset+warmtrain_pts]-esn.prediction_train[:,:])**2)/total_var)
    nrmse_test = np.sqrt(np.mean((mackeyglass_soln[0:1,offset+warmtrain_pts:offset+warmtrain_pts+lyaptime_pts]-prediction[:,0:lyaptime_pts])**2)/total_var)
    
    nrmse_train_statistics[i] = nrmse_train
    nrmse_test_statistics[i] = nrmse_test
    
    print('Run number ' + str(i) + ':')
    print('training NRMSE ESN: '+str(nrmse_train))
    print('testing NRMSE ESN: ' + str(nrmse_test))
    
    plot_results(t_eval, mackeyglass_soln, esn.prediction_train, prediction, offset, warmup, traintime, dt, tau, plottime_tr, plottime_ts, nrmse_train, nrmse_test)


print('Average results:')
print('Mean training NRMSE ESN: '+str(np.mean(nrmse_train_statistics)))
print('Mean testing NRMSE ESN: ' + str(np.mean(nrmse_test_statistics)))




