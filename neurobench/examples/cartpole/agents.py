import torch
from torch.utils.data import DataLoader

import snntorch as snn

from torch import nn
from snntorch import surrogate
from neurobench.models.snntorch_models import SNNTorchAgent
from neurobench.models.torch_model import TorchAgent

import matplotlib.pyplot as plt
from matplotlib import animation

import torch.nn.functional as F

class ActorCriticSNN_LIF_Smallest(torch.nn.Module):
    def __init__(self, num_inputs, action_space, inp_min = torch.tensor([0,-2]), inp_max=  torch.tensor([2,2]),bias=False, nr_passes = 1 ):
        super(ActorCriticSNN_LIF_Smallest, self).__init__()
        self.spike_grad = surrogate.FastSigmoid.apply

        beta = 0.3
        self.nr_passes = nr_passes

        self.lin1 = nn.Linear(num_inputs, 246)
        self.lif1 = snn.Leaky(beta = .75, spike_grad=self.spike_grad, learn_beta=False)

        # basically not spiking final layer
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(246, 1)
        self.lif_critic = snn.Leaky(beta = 0.25, spike_grad=self.spike_grad, learn_beta=False,reset_mechanism='none')

        self.actor_linear = nn.Linear(246, num_outputs)
        self.lif_actor = snn.Leaky(beta = 0.25, spike_grad=self.spike_grad, learn_beta=False, reset_mechanism='none')

     
       # membranes at t = 0
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif_critic.init_leaky()
        self.mem3     = self.lif_actor.init_leaky()

        self.inp_min = inp_min
        self.inp_max = inp_max

        self.train()

        self.inputs = []
        
        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential

        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential


        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential
        
    def reset(self):
        self.inputs = []
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif_critic.init_leaky()
        self.mem3     = self.lif_actor.init_leaky()

        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential

        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential

        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential


    def forward(self, inputs, nr_passes = 1):
        
        # for i in range(self.nr_passes):
        inputs = (inputs - self.inp_min)/(self.inp_max - self.inp_min)

        inputs = inputs.to(torch.float32)

        # use first layer to build up potential and spike and learn weights to present informatino in meaningful way
        cur1 = self.lin1(inputs)
        spk1, self.mem1 = self.lif1(cur1, self.mem1)



        actions =  self.actor_linear(spk1)
        spk_actions,self.mem2 = self.lif_actor(actions, self.mem2)
        val = self.critic_linear(spk1)
        val, self.mem3 = self.lif_critic(val, self.mem3)
        # add information for plotting purposes
        self.inputs.append(inputs.squeeze(0).detach().numpy())
        self.spk_in_rec.append(spk1.squeeze(0).detach())  # Record the output trace of spikes
        self.mem_in_rec.append(self.mem1.squeeze(0).detach().numpy())  # Record the output trace of membrane potential

        self.spk1_rec.append(spk_actions.squeeze(0).detach().numpy())  # Record the output trace of spikes
        self.mem2_rec.append(self.mem2.squeeze(0).detach().numpy())  # Record the output trace of membrane potential
        self.mem3_rec.append(self.mem3.squeeze(0).detach().numpy())  # Record the output trace of membrane potential
        val = self.mem3
        actions = self.mem2
        return val, actions
    

    def plot_spikes(self):
        print("Plotting spikes\n\n\n")
        # print(self.inputs)
        fig, ax = plt.subplots()
        print(self.lin1.weight.data.shape)
        def animate(i):
            ax.clear()
            ax.set_xlim(-1, 250)
            ax.set_ylim(-1, 3)

            # plot input neurons
            for j in range(4):
                ax.add_artist(plt.Circle((j, 0), 0.2, color=plt.cm.Reds(self.inputs[i][j])))

            # plot spikes_1
            for j in range(100):
                ax.add_artist(plt.Circle((j, 1), 0.2, color=plt.cm.Blues(self.spk_in_rec[i][j])))

            # plot spikes_2
            for j in range(2):
                ax.add_artist(plt.Circle((j, 2), 0.2, color=plt.cm.Greens(self.mem2_rec[i][j])))
            
            for j in range(1):
                ax.add_artist(plt.Circle((j, 2.5), 0.2, color=plt.cm.Greens(self.mem3_rec[i][j])))

        ani = animation.FuncAnimation(fig, animate, frames=len(self.inputs), interval=50)
        plt.show()

class ActorCriticSNN_LIF_Small(torch.nn.Module):
    def __init__(self, num_inputs, action_space, inp_min = torch.tensor([0,0]), inp_max=  torch.tensor([1,1]),bias=False, nr_passes = 1 ):
        super(ActorCriticSNN_LIF_Small, self).__init__()
        self.spike_grad = surrogate.FastSigmoid.apply
        num_outputs = action_space.n

        beta = 0.95
        self.nr_passes = nr_passes

        self.lin1 = nn.Linear(num_inputs, 128)
        self.lif1 = snn.Leaky(beta = .95, spike_grad=self.spike_grad, learn_beta=True)
        self.lin2 = nn.Linear(128, 128)
        self.lif2 = snn.Leaky(beta = beta, spike_grad=self.spike_grad, learn_beta=True)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, num_outputs)
     
       # membranes at t = 0
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif2.init_leaky()

        self.inp_min = inp_min
        self.inp_max = inp_max

        self.train()

        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential

        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential


        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential
        
    def init_mem(self):
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif2.init_leaky()

        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential

        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential

        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential


    def forward(self, inputs, nr_passes = 1):
        
        for i in range(self.nr_passes):
            inputs = torch.tensor(inputs).to(torch.float32)
            inputs = (inputs - self.inp_min)/(self.inp_max - self.inp_min)

            inputs = inputs.to(torch.float32)

            # use first layer to build up potential and spike and learn weights to present informatino in meaningful way
            cur1 = self.lin1(inputs)
            spk1, self.mem1 = self.lif1(cur1, self.mem1)

            cur2 = self.lin2(spk1)
            spk2, self.mem2 = self.lif2(cur2, self.mem2)



        actions =  self.actor_linear(spk1)
        
        val = self.critic_linear(spk1)
        
        # add information for plotting purposes
        self.spk_in_rec.append(spk1)  # Record the output trace of spikes
        self.mem_in_rec.append(self.mem1)  # Record the output trace of membrane potential

        self.spk1_rec.append(spk2)  # Record the output trace of spikes
        self.mem1_rec.append(self.mem2)  # Record the output trace of membrane potential


        return val, actions
    
    

class ActorCritic_ANN(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic_ANN, self).__init__()
        self.lin1 = nn.Linear(num_inputs, 128)
        self.lin2 = nn.Linear(128, 128)
 

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, num_outputs)

        self.train()

    def forward(self, inputs):
        inputs = inputs.to(torch.float32)

        x = F.relu(self.lin1(inputs))
        x = F.elu(self.lin2(x))

        return self.critic_linear(x), self.actor_linear(x)
    

class ActorCritic_ANN_Smallest(torch.nn.Module):
    def __init__(self, num_inputs, action_space, hidden_size = 246):
        super(ActorCritic_ANN_Smallest, self).__init__()
        self.lin1 = nn.Linear(num_inputs, hidden_size)
 

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(hidden_size, 1)
        self.actor_linear = nn.Linear(hidden_size, num_outputs)

        self.train()

    def forward(self, inputs):
        inputs = inputs.to(torch.float32)

        x = F.relu(self.lin1(inputs))
        # x = F.elu(self.lin2(x))

        return self.critic_linear(x), self.actor_linear(x)
    

class ActorCriticSNN_LIF_Smallest_pruned(torch.nn.Module):
    def __init__(self, nr_ins, nr_outs,hidden_size=50, inp_min = torch.tensor([0,0]), inp_max=  torch.tensor([1,1]), nr_passes = 1 ):
        super(ActorCriticSNN_LIF_Smallest_pruned, self).__init__()
        self.spike_grad = surrogate.FastSigmoid.apply

        self.nr_ins = nr_ins
        self.nr_outs = nr_outs.n
        self.hidden_size = hidden_size

        beta = 0.3
        self.nr_passes = nr_passes

        self.lin1 = nn.Linear(nr_ins, hidden_size)
        self.lif1 = snn.Leaky(beta = .45, spike_grad=self.spike_grad, learn_beta=True)
        
        # basically not spiking final layer
        num_outputs = nr_outs
        print(hidden_size, num_outputs)
        self.actor_linear = nn.Linear(hidden_size, self.nr_outs)
        self.lif_actor = snn.Leaky(beta = 0, spike_grad=self.spike_grad, learn_beta=False, reset_mechanism='none')

     
       # membranes at t = 0
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif_actor.init_leaky()

        self.inp_min = inp_min
        self.inp_max = inp_max

        self.train()

        self.inputs = []
        
        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential

        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential


        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential
        
    def init_mem(self):
        self.inputs = []
        self.mem1     = self.lif1.init_leaky()
        self.mem2     = self.lif_actor.init_leaky()

        self.spk_in_rec = []  # Record the output trace of spikes
        self.mem_in_rec = []  # Record the output trace of membrane potential

        self.spk1_rec = []  # Record the output trace of spikes
        self.mem1_rec = []  # Record the output trace of membrane potential
        
        self.spk2_rec = []  # Record the output trace of spikes
        self.mem2_rec = []  # Record the output trace of membrane potential

        self.spk3_rec = []  # Record the output trace of spikes
        self.mem3_rec = []  # Record the output trace of membrane potential


    def forward(self, inputs, nr_passes = 1):
        
        for i in range(self.nr_passes):
            inputs = (inputs - self.inp_min)/(self.inp_max - self.inp_min)

            inputs = inputs.to(torch.float32)

            # use first layer to build up potential and spike and learn weights to present informatino in meaningful way
            cur1 = self.lin1(inputs)
            spk1, self.mem1 = self.lif1(cur1, self.mem1)



        actions =  self.actor_linear(spk1)
        spk_actions,self.mem2 = self.lif_actor(actions, self.mem2)

        # add information for plotting purposes
        self.inputs.append(inputs.squeeze(0).detach().numpy())
        self.spk_in_rec.append(spk1.squeeze(0).detach())  # Record the output trace of spikes
        self.mem_in_rec.append(self.mem1.squeeze(0).detach().numpy())  # Record the output trace of membrane potential

        self.spk1_rec.append(spk_actions.squeeze(0).detach().numpy())  # Record the output trace of spikes
        self.mem2_rec.append(self.mem2.squeeze(0).detach().numpy())  # Record the output trace of membrane potential
        actions = self.mem2
        return 0,actions
    

    def plot_spikes(self):
        print("Plotting spikes\n\n\n")
        # print(self.inputs)
        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            ax.set_xlim(-1, self.hidden_size)
            ax.set_ylim(-1, 3)

            # plot input neurons
            for j in range(self.nr_ins):
                ax.add_artist(plt.Circle((j, 0), 0.2, color=plt.cm.Reds(self.inputs[i][j])))

            # plot spikes_1
            for j in range(self.hidden_size):
                ax.add_artist(plt.Circle((j, 1), 0.2, color=plt.cm.Blues(self.spk_in_rec[i][j])))

            # plot spikes_2
            for j in range(self.nr_outs):
                ax.add_artist(plt.Circle((j, 2), 0.2, color=plt.cm.Greens(self.mem2_rec[i][j])))
            
            
        ani = animation.FuncAnimation(fig, animate, frames=len(self.inputs), interval=50)
        plt.show()