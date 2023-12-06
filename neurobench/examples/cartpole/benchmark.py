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

from neurobench.examples.cartpole.agents import ActorCriticSNN_LIF_Smallest, ActorCritic_ANN_Smallest, ActorCriticSNN_LIF_Small, ActorCriticSNN_LIF_Smallest_pruned, ActorCritic_ANN

env = gym.make('CartPole-v0')
env.tau = 0.05
nr_ins = env.observation_space.shape[0]
nr_outs = env.action_space.n


# postprocessors
postprocessors = [discrete_Actor_Critic]

static_metrics = ["model_size", "connection_sparsity"]
data_metrics = ["activation_sparsity", 'reward_score','synaptic_operations']


ANN = ActorCritic_ANN(nr_ins, env.action_space)
ANN.load_state_dict(torch.load('neurobench/examples/cartpole/model_data/ANN_in128x2out_50e3_20hz_lower_entropy_loss.pt'))

model_ann = TorchAgent(ANN)
benchmark = Benchmark_Closed_Loop(model_ann, env, [], postprocessors, [static_metrics, data_metrics])
results = benchmark.run(nr_interactions=1000,max_length=10000)
print(results)

# SNN =  ActorCriticSNN_LIF_Small(nr_ins,env.action_space,
#                                     inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
#                                     inp_max=  torch.tensor([4.8, 10,0.418,2]), 
#                                     bias=False,nr_passes = 1)
# SNN.load_state_dict(torch.load('neurobench/examples/cartpole/model_data/SNN_in128x2out_50e3_20hz_real.pt'))
# print(SNN.lif1.beta)
# print(SNN.lif2.beta)
# model_snn = SNNTorchAgent(SNN)
# benchmark = Benchmark_Closed_Loop(model_snn, env, [], postprocessors, [static_metrics, data_metrics])
# results = benchmark.run(nr_interactions=1000, max_length=10000)
# print(results)
# SNN =  ActorCriticSNN_LIF_Smallest_pruned(nr_ins,env.action_space, hidden_size=11,
#                                     inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
#                                     inp_max=  torch.tensor([4.8, 10,0.418,2]), 
#                                     nr_passes = 1)
# SNN.load_state_dict(torch.load('neurobench/examples/cartpole/model_data/SNN_in248out_25e3_0gain_pruned_full.pt'))

# model_snn = SNNTorchAgent(SNN)

# benchmark = Benchmark_Closed_Loop(model_snn, env, [], postprocessors, [static_metrics, data_metrics])
# results = benchmark.run(nr_interactions=1000, max_length=10000)
# print(results)