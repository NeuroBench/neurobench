import numpy as np
import torch

import csv

OPS_SEED = 1337
SIMULATOR_SEED = 345783

def normalize(vector):
    mag = torch.linalg.norm(vector)

    if (mag == 0):
        return vector
    else:
        return vector / mag

class SyntheticNeuron():
    def __init__(
        self,
        time_step: float,
        upper_lmax: float,
        lower_lmax: float,
        upper_lmin: float,
        max_accel: float,
        seed: int,
        zero_prob: float=0,
        device: str='cpu'
    ):
        """
        Args:
            time_step: The time taken per iteration (s).
            upper_lmax: upper limit of the maximum firing rate of a neuron
            lower_lmax: lower limit of the maximum firing rate of a neuron
            upper_lmin: upper limit of the minimum firing rate of a neuron
            max_accel: limits the acceleration of the neuron
            seed: random seed for the neuron
            zero_prob: probability of the neuron producing no spikes
            device: device to run the neuron on (cpu or cuda)
        """
        self.rng_manager = torch.Generator()
        self.rng_manager.manual_seed(seed)

        self.max_accel = max_accel
        self.time_step = time_step

        self.device = device
        
        probs = torch.tensor([zero_prob, 1 - zero_prob])
        zero_choice = torch.multinomial(probs, 1, generator=self.rng_manager).cpu().item()
        self.lambda_min = torch.empty(1).uniform_(0, upper_lmin, generator=self.rng_manager).cpu().item() * zero_choice

        self.lambda_max = torch.empty(1).uniform_(max(self.lambda_min, lower_lmax), upper_lmax, generator=self.rng_manager).cpu().item()

        self.theta_prefer = torch.empty(1).uniform_(-torch.pi, torch.pi, generator=self.rng_manager).cpu().item()
        self.c = torch.tensor([np.cos(self.theta_prefer), np.sin(self.theta_prefer)], dtype=torch.float)

        self.removed = False

    def assign(self,c,lambda_min,lambda_max):
        self.c = c
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def get_spike(self,v_t):
        inner_prod = torch.min(torch.ones(1), torch.max(torch.zeros(1), 1.5*torch.inner(self.c.cpu(), v_t.cpu())) / self.max_accel)
        
        lambda_t = (self.lambda_max-self.lambda_min)*inner_prod + self.lambda_min

        p = lambda_t * self.time_step
        if self.removed:
            p = 0.0

        probs = torch.tensor([1-p, p])

        return torch.multinomial(probs, 1, generator=self.rng_manager).item()

class OPS():
    def __init__(
        self,
        num_neurons: int,
        time_step: float,
        upper_lmax: float,
        lower_lmax: float,
        upper_lmin: float,
        zero_prob: float=0,
        device: str='cpu'
    ):
        """
        Args:
            num_neurons: Number of neurons used by the OPS.
            time_step: The time taken per iteration (s).
            upper_lmax: upper limit of the maximum firing rate of a neuron
            lower_lmax: lower limit of the maximum firing rate of a neuron
            upper_lmin: upper limit of the minimum firing rate of a neuron
            zero_prob: probability of the neuron producing no spikes
            device: device to run the OPS on (cpu or cuda)
        """
        
        self.rng_manager = torch.Generator(device=device)
        self.rng_manager.manual_seed(OPS_SEED)
        self.synth_seed_list = torch.randperm(num_neurons, generator=self.rng_manager, device=device).cpu().tolist()

        self.num_neurons = num_neurons
        self.time_step = time_step
        self.max_velocity = 20*time_step
        self.accel_const = 0.3
        self.max_accel = self.max_velocity * self.accel_const

        self.device = device

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(SyntheticNeuron(
                time_step,
                upper_lmax,
                lower_lmax,
                upper_lmin,
                self.max_accel,
                seed=self.synth_seed_list[i],
                zero_prob=zero_prob,
                device=self.device
            ))

    def get_spikes(self,v_t):
        spikes= torch.zeros((self.num_neurons,), device=self.device)
        for i in range(self.num_neurons):
            spikes[i] = self.neurons[i].get_spike(v_t)
        return spikes

    def remove_neurons(self,indices):
        for i in indices:
            self.neurons[i].removed = True

    def save_neurons(self,filename):
        with open(filename,'w') as file:
            writer = csv.writer(file)
            for i in range(len(self.neurons)):
                neuron = self.neurons[i]
                writer.writerow([neuron.c[0].item(),neuron.c[1].item(),neuron.lambda_min,neuron.lambda_max])

    def assign_neurons(self, filename):
        self.neurons = []
        with open(filename,'r') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                c = torch.tensor([float(row[0]), float(row[1])])
                lambda_min = float(row[2])
                lambda_max = float(row[3])

                neuron = SyntheticNeuron(self.time_step,upper_lmin=5,lower_lmax=40,upper_lmax=100,max_accel=self.max_accel, seed=self.synth_seed_list[i],zero_prob=0.,device=self.device)
                neuron.assign(c, lambda_min, lambda_max)
                self.neurons.append(neuron)

        self.num_neurons = len(self.neurons)

class OPSEnv():
    def __init__(
        self,
        ops: OPS,
        max_duration: float,
        min_time_in_target: float=0.5,
        side_radius=10,
        min_distance=8,
        target_size=1, 
        device='cpu'
    ):
        """
        Args:
            ops: The OPS object that governs how the state updates.
            max_duration: The maximum duration of the simulation (s).
            min_time_in_target: min time for the model output to stay in target area
            side_radius: radius of the OPS environment.
            min_distance: minimum distance to the target from the origin.
            target_size: size of the target area.
            device: device to run the OPS on (cpu or cuda)
        """
        self.rng_manager = torch.Generator()
        self.rng_manager.manual_seed(SIMULATOR_SEED)

        self.ops = ops

        self.t = 0
        self.max_duration = max_duration
        self.min_time_in_target = min_time_in_target
        self.time_in_range = 0

        self.vel_scale = 3.77
        self.side_radius = side_radius
        self.max_vel = ops.max_velocity
        self.accel_const = ops.accel_const
        self.target = [0.0,0.0]
        self.target_size = target_size
        self.min_distance = min_distance
        self.distance_const = self.max_vel / np.sqrt(2*self.target_size) # Velocity peaks when you're half the side length away from an object
        self.terminate = False

        self.device = device

    def reward_fn(self):
        return 1

    def reset(self):
        target_mag = self.min_distance
        target_angle = torch.empty(1).uniform_(-torch.pi, torch.pi, generator=self.rng_manager).item()

        self.target = target_mag * torch.tensor([np.cos(target_angle), np.sin(target_angle)], dtype=torch.float, device=self.device)
        self.position = torch.tensor([0.0, 0.0], dtype=torch.float, device=self.device)
        self.velocity = torch.tensor([0.0, 0.0], dtype=torch.float, device=self.device)
        self.t = 0
        self.time_in_range = 0
        self.terminate = False

        new_velocity, delta_velocity = self.get_velocity()
        spikes = self.ops.get_spikes(delta_velocity)

        return spikes, None

    def step(self, model_output):
        self.update_pos(model_output * self.vel_scale * self.ops.time_step)
        new_velocity, delta_velocity = self.get_velocity()
        spikes = self.ops.get_spikes(delta_velocity)

        vels = new_velocity / self.ops.time_step
        vels /= self.vel_scale

        if self.t*self.ops.time_step < self.max_duration and self.time_in_range*self.ops.time_step < self.min_time_in_target:
            self.terminate = False
        else:
            self.terminate = True

        return spikes, self.reward_fn(), self.terminate, None, None

    def get_velocity(self):
        vector = self.target - self.position

        angle = normalize(vector)
        vel_mag = min(self.distance_const*np.sqrt(np.linalg.norm(vector.cpu())), self.max_vel)
        new_velocity = vel_mag*angle
        
        delta_velocity = (new_velocity - self.velocity)*self.accel_const
        #The acceleration constant prevents instantaneous jumps in velocity

        new_velocity =  self.velocity + delta_velocity

        return new_velocity, delta_velocity
    
    def update_pos(self, new_vel):
        self.position += new_vel
        self.velocity = new_vel
        self.t += 1
        target_dist = np.linalg.norm(self.position.cpu() - self.target.cpu())

        if (target_dist < self.target_size):
            self.time_in_range += 1
        else:
            self.time_in_range = 0


