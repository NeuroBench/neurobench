from neurobench.datasets.dataset import Dataset
import numpy as np
import torch
import math
from jitcdde import jitcdde, y, t, jitcdde_lyap


class MackeyGlass(Dataset):
    """ Dataset for the Mackey-Glass task.
    """
    def __init__(self, 
                 tau,  
                 constant_past,
                 nmg = 10, 
                 beta = 0.2, 
                 gamma = 0.1,
                 dt=1.0, 
                 splits=(8000., 2000.),
                 start_offset=0.,
                 seed_id=0,
    ):
        """
        Initializes the Mackey-Glass dataset.

        Args:
            tau (float): parameter of the Mackey-Glass equation
            constant_past (float): initial condition for the solver
            nmg (float): parameter of the Mackey-Glass equation
            beta (float): parameter of the Mackey-Glass equation
            gamma (float): parameter of the Mackey-Glass equation
            dt (float): time step length for sampling data
            splits (tuple): data split in time units for training and testing data, respectively
            start_offset (float): added offset of the starting point of the time-series, in case of repeating using same function values
            seed_id (int): seed for generating function solution
        """

        super().__init__()

        # Parameters
        self.tau = tau
        self.constant_past = constant_past
        self.nmg = nmg
        self.beta = beta
        self.gamma = gamma
        self.dt = dt
        
        # Time units for train (user should split out the warmup or validation)
        self.traintime = splits[0]
        # Time units to forecast
        self.testtime = splits[1]
        
        self.start_offset = start_offset
        self.seed_id = seed_id

        # Total time to simulate the system
        self.maxtime = self.traintime + self.testtime + self.dt

        # Discrete-time versions of the continuous times specified above
        self.traintime_pts = round(self.traintime/self.dt)
        self.testtime_pts = round(self.testtime/self.dt)
        self.maxtime_pts = self.traintime_pts + self.testtime_pts + 1 # eval one past the end

        # Specify the system using the provided parameters
        self.mackeyglass_specification = [ self.beta * y(0,t-self.tau) / (1 + y(0,t-self.tau)**self.nmg) - self.gamma*y(0) ]

        # Generate time-series
        self.generate_data()

        # Generate train/test indices
        self.split_data()
        

    def generate_data(self):
        """ Generate time-series using the provided parameters of the equation.
        """
        np.random.seed(self.seed_id)

        # Create the equation object based on the settings
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
        for time in torch.arange(self.DDE.t+self.start_offset, self.DDE.t+self.start_offset+self.maxtime, self.dt,dtype=torch.float64):
            value, lyap, weight = self.DDE.integrate(time.item())
            self.mackeyglass_soln[count,0] = value[0] 
            lyaps[count,0] = lyap[0]
            lyaps_weights[count,0] = weight  
            count += 1

        # Total variance of the generated Mackey-Glass time-series
        self.total_var=torch.var(self.mackeyglass_soln[:,0], True)
        
        # Estimate Lyapunov exponent
        self.lyap_exp = ((lyaps.T@lyaps_weights)/lyaps_weights.sum()).item()


    def split_data(self):
        """ Generate training and testing indices.
        """
        self.ind_train = torch.arange(0, self.traintime_pts)
        self.ind_test = torch.arange(self.traintime_pts, self.maxtime_pts-1)
             
    def __len__(self):
        """ Returns number of samples in dataset.

        Returns:
            int: number of samples in dataset
        """
        return len(self.mackeyglass_soln)-1
    
    def __getitem__(self, idx):
        """ Getter method for dataset.

        Args:
            idx (int): index of sample to return

        Returns:
            sample (tensor): individual data sample, shape=(timestamps, features)=(1,1)
            target (tensor): corresponding next state of the system, shape=(label,)=(1,)
        """
        sample = torch.unsqueeze(self.mackeyglass_soln[idx, :], dim=0)
        target = self.mackeyglass_soln[idx+1, :]

        return sample, target
