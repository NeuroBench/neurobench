import torch
from torch.utils.data import DataLoader

import snntorch as snn

from torch import nn
from snntorch import surrogate

import gym
from neurobench.models.snntorch_models import SNNTorchAgent
from neurobench.models.torch_model import TorchAgent
from neurobench.benchmarks import Benchmark_Closed_Loop
from neurobench.examples.cartpole.processors import discrete_Actor_Critic

from neurobench.examples.cartpole.agents import ActorCriticSNN_LIF_Smallest, ActorCritic_ANN_Smallest

env = gym.make('CartPole-v0')
nr_ins = env.observation_space.shape[0]
nr_outs = env.action_space.n
ANN = ActorCritic_ANN_Smallest(nr_ins, env.action_space)
ANN.load_state_dict(torch.load('neurobench/examples/cartpole/model_data/ANN_in248out_50e3_0gain.pt'))
model_ann = TorchAgent(ANN)

SNN =  ActorCriticSNN_LIF_Smallest(nr_ins,env.action_space,
                                    inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
                                    inp_max=  torch.tensor([4.8, 10,0.418,2]), 
                                    bias=False,nr_passes = 1)
SNN.load_state_dict(torch.load('neurobench/examples/cartpole/model_data/SNN_in248out_50e3_0gain.pt'))
model_snn = SNNTorchAgent(SNN)

# postprocessors
postprocessors = [discrete_Actor_Critic]

static_metrics = ["model_size", "connection_sparsity"]
data_metrics = ["activation_sparsity", 'reward','average_time']

benchmark = Benchmark_Closed_Loop(model_ann, env, [], postprocessors, [static_metrics, data_metrics])
results = benchmark.run()
print(results)

benchmark = Benchmark_Closed_Loop(model_snn, env, [], postprocessors, [static_metrics, data_metrics])
results = benchmark.run()
print(results)