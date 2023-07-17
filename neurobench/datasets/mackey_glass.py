"""
=====================================================================
Project:      NeuroBench
File:         mackey_glass.py
Description:  Python code describing dataloader for the Mackey-Glass task
Date:         14. July 2023
=====================================================================
Copyright stuff
=====================================================================
"""


from neurobench.datasets.dataset import Dataloader
import numpy as np
import torch
import math
from jitcdde import jitcdde, y, t, jitcdde_lyap


class MackeyGlass(Dataloader):
    """
    Dataloader for the Mackey-Glass task

    Parameters
    ----------
    repeat  : int
        number of runs a model will be evaluted for
    tau    :  float
        parameter of the Mackey-Glass equation
    constant_past    :  float
        initial condition for the solver
    nmg    :  float
        parameter of the Mackey-Glass equation
    beta    :  float
        parameter of the Mackey-Glass equation
    gamma    :  float
        parameter of the Mackey-Glass equation
    dt  : float
        time step for sampling data
    splits  : list
        data split in time units for warmup, training, and testing data, respectively
    runspantime  : float
        units of time to to separate adjacent time-series for repeats
    seed  : int
        seed of NumPy preuso-random generator to genetate determenistic time-series 

    Methods
    ----------
    generate__data
        generate time-series using the provided parameters of the equation 
    split_data
        split data into warmup+training and testing sets
    """
    def __init__(self, 
                 repeat,
                 tau,  
                 constant_past,
                 nmg = 10, 
                 beta = 0.2, 
                 gamma = 0.1,
                 dt=1.0, 
                 splits=(1000., 7000., 2000.),
                 runspantime = 1000.,
                 seed = 0,
    ):

        super().__init__()

        # Parameters
        self.repeat = repeat
        self.tau = tau
        self.constant_past = constant_past
        self.nmg = nmg
        self.beta = beta
        self.gamma = gamma
        self.dt = dt
        # Units of time for pre-training warm-up period
        self.warmup = splits[0]
        # Units of time to train for
        self.traintime = splits[1]
        # Units of time to forecast for
        self.testtime = splits[2]
        self.runspantime = runspantime 
        self.seed = seed
        
        # Total time to run for in a single simulation
        self.singleruntime = self.warmup+self.traintime+self.testtime
        # Total time to simulate the system
        self.maxtime = (self.repeat-1)*self.runspantime + self.singleruntime
        # Discrete-time versions of the continuous times specified above
        self.warmup_pts=round(self.warmup/self.dt)
        self.traintime_pts=round(self.traintime/self.dt)
        self.warmtrain_pts=self.warmup_pts+self.traintime_pts
        self.testtime_pts=round(self.testtime/self.dt)
        self.runspantime_pts = math.floor(self.runspantime/self.dt)
        self.singleruntime_pts = round(self.singleruntime/self.dt)
        self.t_eval = torch.arange(0, self.maxtime, self.dt,dtype=torch.float64)
        self.maxtime_pts=len(self.t_eval)

        # Specify the system using the provided parameters
        self.mackeyglass_specification = [ self.beta * y(0,t-self.tau) / (1 + y(0,t-self.tau)**self.nmg) - self.gamma*y(0) ]

        # Generate time-series
        self.generate_data()

    def generate_data(self):
        """
        Generate time-series using the provided parameters of the equation 

        """

        # Create the equation object based on the settings
        np.random.seed(seed=self.seed)
        self.DDE = jitcdde_lyap(self.mackeyglass_specification)
        self.DDE.constant_past([self.constant_past])
        self.DDE.step_on_discontinuities()

        ##
        ## Generate data from the Mackey-Glass system
        ##
        self.mackeyglass_soln = torch.zeros((self.maxtime_pts,1),dtype=torch.float64)
        lyaps = torch.zeros((self.maxtime_pts,1),dtype=torch.float64)
        lyaps_weights = torch.zeros((self.maxtime_pts,1),dtype=torch.float64)
        count = 0
        for time in torch.arange(self.DDE.t, self.DDE.t+self.maxtime, self.dt,dtype=torch.float64):
            value, lyap, weight = self.DDE.integrate(time.item())
            self.mackeyglass_soln[count,0] = value[0] 
            lyaps[count,0] = lyap[0]
            lyaps_weights[count,0] = weight  
            count += 1

        # Rotal variance of the generated Mackey-Glass time-series
        self.total_var=torch.var(self.mackeyglass_soln[:,0], True)
        
        # Estimate Lyapunov exponent
        self.lyap_exp = ((lyaps[self.warmup_pts:].T@lyaps_weights[self.warmup_pts:])/lyaps_weights[self.warmup_pts:].sum()).item()


    def split_data(self, idx):
        """
        Split data into warmup+training and testing sets according to the time steps specified in self.warmtrain_pts and self.testtime_pts.
        
        Parameters
        ----------
        idx : int
            index of the repeat; chooses the data using self.runspantime_pts

        """
        # Compute offset for the current idx repeat 
        self.offset = idx*self.runspantime_pts
        
        # Extract data based on offset and length of the corresponding sequences
        self.training_data = self.mackeyglass_soln[self.offset:self.offset+self.warmtrain_pts,:]   
        self.training_targets = self.mackeyglass_soln[self.offset+self.warmup_pts+1:self.offset+self.warmtrain_pts+1,:]
        self.test_data = self.mackeyglass_soln[self.offset+self.warmtrain_pts:self.offset+self.singleruntime_pts,:]  
             
        