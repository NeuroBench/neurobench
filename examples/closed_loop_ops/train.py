import numpy as np
import time, csv
import copy

import torch
import torch.nn as nn

import time
import random
import os
from tqdm import tqdm

from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager

import sys
sys.path.append("/home/vsun/closed_loop_test/")
from neurobench.envs import OPS, OPSEnv
from neurobench.models.torch_model import TorchModel
from neurobench.benchmarks import BenchmarkClosedLoop

from model import ANNModel

random_seed = 1234 #5678
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(random_seed)

def get_trial(cls, max_duration, time_to_target):
    cls.reset()
    vels = torch.zeros((max_duration,2), device=cls.device)
    accels = torch.zeros((max_duration,2), device=cls.device)
    t = 0
    t_in_range = 0
    while cls.t < max_duration and cls.time_in_range < time_to_target:
        vels[t,:],accels[t,:] = cls.get_velocity()
        mag = np.linalg.norm(vels[t,:])
        cls.update_pos(vels[t,:] + np.random.normal(loc=0,scale=0.1*mag,size=(2,)))
        # t, t_in_range = cls.get_times()

    vels = vels[:cls.t,:]
    accels = accels[:cls.t,:]
    
    return vels,accels

def get_spikes(ops,accels):
    spikes = np.zeros((len(accels),ops.num_neurons))
    for t in range(len(accels)):
        spikes[t,:] = ops.get_spikes(accels[t])

    return spikes

time_step = 0.01
num_neurons=96
max_duration = 3.0
min_time_in_target = 0.5 # value in seconds
target_size = 2.5

model_weight_name = "./" + "OPS_" + "model_state_dict.pth" 

ops = OPS(
    num_neurons=num_neurons,
    time_step=time_step,
    upper_lmax=100,
    lower_lmax=40,
    upper_lmin=5,
    zero_prob=0.5,
)

env = OPSEnv(
    ops=ops,
    max_duration=max_duration,
    min_time_in_target=min_time_in_target,
    side_radius=10,
    min_distance=0,
    target_size=target_size,
)

vels_train = torch.zeros((1,2))
spikes_train = torch.zeros((1,num_neurons))
vels_test = torch.zeros((1,2))
spikes_test = torch.zeros((1,num_neurons))

train_trials = 400

for trial in tqdm(range(train_trials+1), desc="aj;dsfis"):
    vels,accels = get_trial(env, int(env.max_duration), env.min_time_in_target)
    vels_train = np.concatenate((vels_train,vels),axis=0)
    spikes = get_spikes(ops,accels)
    spikes_train = np.concatenate((spikes_train,spikes),axis=0)
    
print("Start training")
model = ANNModel(input_dim=num_neurons)

vels_train = vels_train / time_step
vels_train_scaled = vels_train / env.vel_scale

samples = torch.tensor(spikes_train, dtype=torch.float32)
labels = torch.tensor(vels_train_scaled, dtype=torch.float32)

criterion = torch.nn.MSELoss()
optimiser = torch.optim.AdamW(model.parameters(), lr=0.005, 
                                betas=(0.9, 0.999), weight_decay=0.05)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiser, T_max=10)
for epoch in tqdm(range(50)):
    # model.to('cuda')
    model.train()

    for i in range(0, samples.shape[0]-128, 128):
        
        label = labels[i:i+128, :].squeeze()
        sample = samples[i:i+128, :]

        pred = model.forward(sample)
        
        loss_val = criterion(pred, label)

        optimiser.zero_grad()
        loss_val.backward()
        optimiser.step()

    lr_scheduler.step()

torch.save(model.state_dict(), model_weight_name)
