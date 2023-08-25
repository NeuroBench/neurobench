"""
=====================================================================
Project:      NeuroBench
File:         echo_state_network.py
Description:  Python code providing an implementation of Echo State Networks 
Date:         20. July 2023
=====================================================================
Copyright stuff
=====================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EchoStateNetwork(nn.Module):
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
        mode: str = "autonomous"    
    ):
        super(EchoStateNetwork, self).__init__()
        self.in_channels = in_channels
        self.reservoir_size = reservoir_size
        self.input_scale = input_scale
        self.connect_prob = connect_prob
        self.spectral_radius = spectral_radius
        self.leakage = leakage
        self.ridge_param = ridge_param
        self.include_bias = include_bias
        self.include_input = include_input
        self.mode = mode

        assert self.mode in ["autonomous", "single_step"]
        self.prior_prediction = None
        
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
        torch.manual_seed(seed_id)
        
        #Uniform random projection matrix for feeding input into the reservoir   
        Win = (torch.rand((self.reservoir_size,self.in_channels_bias),dtype=torch.float64)-0.5) 
        #Scaling Win by input_scale
        Win = self.input_scale*Win

        #Recurrent connectivity matrix W for the reservoir
        #Choose nonzero connections 
        W = (torch.rand((self.reservoir_size,self.reservoir_size),dtype=torch.float64) <= self.connect_prob).type(torch.float64) 
        #Assign random values to these connections
        W[W == True] = torch.normal(0.,1., size=(torch.sum(W).type(torch.int).item(),),dtype=torch.float64)
        # Scale the resuling recurrent connectivity matrix 
        w, _ = torch.linalg.eig(W)
        W = self.spectral_radius*W/torch.abs(w[0])
        # Alternative way to form the orthonormal recurrent connectivity matrix
        #self.W, _ = np.linalg.qr(np.random.normal(size=(self.reservoir_size,self.reservoir_size)))
        #self.W = self.spectral_radius*self.W            
        
        self.Win = nn.Linear(self.in_channels_bias, self.reservoir_size, bias = False, dtype = torch.float64)
        self.W = nn.Linear(self.reservoir_size, self.reservoir_size, bias = False, dtype = torch.float64)
        with torch.no_grad():
            self.Win.weight.copy_(Win)
            self.W.weight.copy_(W)
            
    # Performs the training phase of the Echo State Network.
    def fit(
        self, 
        training_data: torch.tensor, 
        targets: torch.tensor, 
        warmup_pts: int,
    ):
        training_data = training_data.view(training_data.numel(), 1)

        warmtrain_pts = training_data.shape[0]
        
        # Form reservoir states for the warm-up and training data
        # Pick the warm-up and training data
        # Add constant bias if applicable
        if self.include_bias:
            training_data = torch.concatenate((1*torch.ones((warmtrain_pts,1)), training_data, ), axis=1)
   
        # Initialize reservoir's state to an empty state
        self.reservoir = torch.zeros((self.reservoir_size, 1),dtype=torch.float64)

        # Obtain the reservoir states for the training phase
        self.reservoir_tr = torch.zeros((self.reservoir_size+self.reservoir_extension, warmtrain_pts),dtype=torch.float64)
        for i in range(0,warmtrain_pts):    
            
            # Project input to the reservoir & Update the reservoir            
            x = torch.tanh(self.W(self.reservoir.T) + self.Win(training_data[i:i+1,:])) 
            self.reservoir = (1-self.leakage)*self.reservoir + self.leakage*x.T            

            # Bias & the current input if applicable
            if self.include_input:
                self.reservoir_tr[:,i:i+1] = torch.cat((training_data[i:i+1,:].T ,self.reservoir), dim=0)                
            else:
                self.reservoir_tr[:,i:i+1] = self.reservoir
    
        # Ridge regression: train the readout matrix Wout to map reservoir_tr to targets. The warmup period defined by warmup_pts is ignored
        #self.Wout = targets @ self.reservoir_tr[:,warmup_pts:].T @ np.linalg.pinv(self.reservoir_tr[:,warmup_pts:] @ self.reservoir_tr[:,warmup_pts:].T + self.ridge_param*np.identity(self.reservoir_size+self.reservoir_extension))
        Wout = torch.linalg.lstsq((self.reservoir_tr[:,warmup_pts:] @ self.reservoir_tr[:,warmup_pts:].T + self.ridge_param*torch.eye(self.reservoir_size+self.reservoir_extension)), (self.reservoir_tr[:,warmup_pts:] @ targets), rcond=None, driver='gelsd')[0].T
        
        self.Wout = nn.Linear(Wout.size(1), Wout.size(0), bias = False, dtype = torch.float64)
        with torch.no_grad():
            self.Wout.weight.copy_(Wout)        
        
        #Predictions on the traiing data
        self.prediction_train =  self.Wout(self.reservoir_tr[:,warmup_pts:].T)

    def single_forward(self, sample):
        # Update the reservoir state for the next predicition 
        if self.include_bias:
            sample_b = torch.concatenate((1*torch.ones((1, 1)), sample,), axis=0)
        else:
            sample_b = sample       

        # Project input to the reservoir & Update the reservoir
        x = torch.tanh(self.W(self.reservoir.T) + self.Win(sample_b.T)) 
        self.reservoir = (1-self.leakage)*self.reservoir + self.leakage*x.T
        
        # Include input if applicable
        if self.include_input:
            x = torch.cat((sample_b,self.reservoir), dim=0)
        else:
            x = self.reservoir
                
        # Make predictions based on the current reservoir state
        prediction = self.Wout(x.T)

        return prediction

    ##
    ## Forecast with ESN for a batch of inputs
    ##     
    def forward(self, batch): # forward is not called during the model fitting
        predictions = []
        for sample in batch:
            if self.mode == 'autonomous' and self.prior_prediction is not None:
                sample = self.prior_prediction
            prediction = self.single_forward(sample)
            predictions.append(prediction)
            self.prior_prediction = prediction

        # reset so that next batch will not have prior prediction
        self.prior_prediction = None

        return torch.tensor(predictions).unsqueeze(-1)
   
        