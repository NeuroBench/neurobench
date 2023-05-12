'''
These are helper functions for the SNN taken from the paper
"An Energy-Efficient Spiking Neural Network for Finger Velocity Decoding for Implantable Brain-Machine Interface."
This code is with minor modifications available at "https://github.com/liaoRichard/SNN-for-Finger-Velocity-iBMI"

=====================================================================
Project:      An Energy-Efficient Spiking Neural Network for Finger Velocity Decoding for Implantable Brain-Machine Interface
File:         layers.py
Description:  Python code describing the individual layers and their training of the SNN
Date:        10. April 2022
=====================================================================
Copyright (C) 2022 ETH Zurich.
Author: Lars Widmer
SPDX-License-Identifier: Apache-2.0
Licensed under the Apache License, Version 2.0 (the License); you may
not use this file except in compliance with the License.
You may obtain a copy of the License at
www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an AS IS BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Notice: The work in this file is partially based on
"STBP-train-and-compression" by liukuang, which is licensed
under the MIT License.
Please see the File "LICENCE.md" for the full licensing information.
=====================================================================
'''

import torch
import torch.nn as nn

def init_surrogate_gradient(hyperparam):
    global surrogate_gradient
    surrogate_gradient = hyperparam['surrogate_gradient']


class SurrogateGradient(torch.autograd.Function):
    """
        Implementation of the spiking activation function with an approximation of gradient.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = torch.gt(input, 0.0)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # hu is an approximate func of df/du
        aa = 0.5
        if (surrogate_gradient == 'square'):
            hu = abs(input) < aa
            hu = hu.float() / (2 * aa)
        else:
            print('no gradient function specified')
        return grad_input * hu


SurrogateGradient = SurrogateGradient.apply


def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n, Vth_, tau_, reset_by_subtraction):
    if reset_by_subtraction:
        u_t1_n1 = tau_ * (
                    u_t_n1 - Vth_ * o_t_n1) + W_mul_o_t1_n  # next voltage = tau*(old_voltage-old outputs*Vth)+inputs (subtract old outputs)
    else:
        u_t1_n1 = tau_ * u_t_n1 * (
                    1 - o_t_n1) + W_mul_o_t1_n  # next voltage = tau*old_voltage*(1-old outputs)+inputs  (reset voltage)

    o_t1_n1 = SurrogateGradient(u_t1_n1 - Vth_)  # next outputs
    return u_t1_n1, o_t1_n1


def state_update_no_spike(u_t_n1, o_t_n1, W_mul_o_t1_n, tau_):
    u_t1_n1 = tau_ * u_t_n1 + W_mul_o_t1_n  # next voltage = tau*old_voltage+inputs
    return u_t1_n1, u_t1_n1


class tdLayer(nn.Module):
    """
        Converts a common layer to the time domain. The input tensor needs to have an additional time dimension, which in this case is on the
         last dimension of the data. When forwarding, a normal layer forward is performed for each time step of the data in that time dimension.
    Args:
        layer (nn.Module):
            The layer needs to convert.
        bn (nn.Module):
            If batch-normalization is needed, the BN layer should be passed in together as a parameter.
    """

    def __init__(self, layer, bn=None, hyperparam=None):
        super(tdLayer, self).__init__()
        self.layer = layer
        self.bn = bn
        self.hyperparam = hyperparam

    def forward(self, x):
        steps = x.shape[-1]
        x_ = torch.zeros(self.layer(x[..., 0]).shape + (steps,), device=x.device)
        for step in range(steps):
            x_[..., step] = self.layer(x[..., step])
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class LIFSpike(nn.Module):
    """
        Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor
        needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """

    def __init__(self, inpshape=None, hyperparam=None):
        super(LIFSpike, self).__init__()
        Vth = hyperparam['Vth']
        tau = hyperparam['tau']
        tau_range = hyperparam['tau_init_range']
        self.inpshape = inpshape
        self.reset_by_subtraction = hyperparam['reset_by_subtraction']
        self.hyperparam = hyperparam

        self.Vth = torch.Tensor(*inpshape).to(hyperparam['device'])
        nn.init.uniform_(self.Vth, Vth, Vth)
        if (hyperparam['tau_trainable']):
            self.tau = nn.Parameter(torch.Tensor(*inpshape)).to(hyperparam['device'])
            nn.init.uniform_(self.tau, tau - tau_range / 2, tau + tau_range / 2)
        else:
            self.tau = torch.Tensor(*inpshape).to(hyperparam['device'])
            nn.init.uniform_(self.tau, tau, tau)

    def forward(self, x):
        steps = x.shape[-1]
        if (self.hyperparam['init_u'] == 'random'):
            u = torch.rand(x.shape[:-1], device=x.device) * self.Vth  # or maybe better to initialize high?
        elif (self.hyperparam['init_u'] == 'zero'):
            u = torch.zeros(x.shape[:-1], device=x.device)
        elif (self.hyperparam['init_u'] == 'Vth'):
            u = torch.ones(x.shape[:-1], device=x.device) * (self.Vth - 1e-6)
        out = torch.zeros(x.shape, device=x.device)

        # constrain tau and Vth
        if (self.hyperparam['constrain_method'] == 'forward'):
            Vth = torch.clamp(self.Vth, min=0.0)
            tau = torch.clamp(self.tau, min=0.0, max=1.0)
        elif (self.hyperparam['constrain_method'] == 'always'):
            self.Vth.data = torch.clamp(self.Vth, min=0.0)
            Vth = self.Vth
            self.tau.data = torch.clamp(self.tau, min=0.0, max=1.0)
            tau = self.tau
        elif (self.hyperparam['constrain_method'] == 'none' or self.hyperparam['constrain_method'] == 'eval'):
            Vth = self.Vth
            tau = self.tau

        for step in range(steps):
            # max(step-1,0) since step is 0 in the first iteration and we need to give the spikes from the last timestep, we give zero instead.
            u, out[..., step] = state_update(u, out[..., max(step - 1, 0)], x[..., step], Vth, tau,
                                             self.reset_by_subtraction)
        return out


class LI_no_Spike(nn.Module):
    """
        Generates . It can be considered as an activation function and is used similar to ReLU. The input tensor
        needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """

    def __init__(self, inpshape=None, hyperparam=None):
        super(LI_no_Spike, self).__init__()
        tau = hyperparam['tau']
        tau_range = hyperparam['tau_init_range']
        self.inpshape = inpshape
        self.hyperparam = hyperparam
        self.Vth = None
        if (hyperparam['tau_trainable']):
            self.tau = nn.Parameter(torch.Tensor(inpshape)).to(hyperparam['device'])
            nn.init.uniform_(self.tau, tau - tau_range / 2, tau + tau_range / 2)
        else:
            self.tau = torch.Tensor(inpshape).to(hyperparam['device'])
            nn.init.uniform_(self.tau, tau, tau)

    def forward(self, x):
        steps = x.shape[-1]
        u = torch.zeros(x.shape[:-1], device=x.device)
        out = torch.zeros(x.shape, device=x.device)

        # constrain tau and Vth
        if (self.hyperparam['constrain_method'] == 'forward'):
            tau = torch.clamp(self.tau, min=0.0, max=1.0)
        elif (self.hyperparam['constrain_method'] == 'always'):
            self.tau.data = torch.clamp(self.tau, min=0.0, max=1.0)
            tau = self.tau
        elif (self.hyperparam['constrain_method'] == 'none' or self.hyperparam['constrain_method'] == 'eval'):
            tau = self.tau
        for step in range(steps):
            u, out[..., step] = state_update_no_spike(u, out[..., max(step - 1, 0)], x[..., step], tau)
        return out


class tdBatchNorm2d(nn.BatchNorm2d):
    """tdBN ?https://arxiv.org/pdf/2011.05280
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well
        when doing BN. Expects inputs and outputs as (Batch, Channel, Height, Width, Time)
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True, Vth=0.0):
        super(tdBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.Vth = Vth
        self.deactivated = False

    def forward(self, input):
        if (self.deactivated):
            return input
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3, 4])
            # use biased var in train
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * self.Vth * (input - mean[None, :, None, None, None]) / (
            torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]

        return input


class tdBatchNorm1d(nn.BatchNorm1d):
    """1d version of tdBatchNorm2d: Only difference is the input and output shapes: Expects inputs and outputs as (Batch, Channel, Width, Time)
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True, Vth=0.0,
                 device='cpu'):
        super(tdBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, device=device)
        self.alpha = alpha
        self.Vth = Vth
        self.deactivated = False

    def forward(self, input):
        if (self.deactivated):
            return input
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * self.Vth * (input - mean[None, :, None, None]) / (
            torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


class tdBatchNorm0d(nn.BatchNorm1d):
    """0d version of tdBatchNorm2d: Only difference is the input and output shapes: Expects inputs and outputs as (Batch, Channel, Time)
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True, Vth=0.0,
                 device='cpu'):
        super(tdBatchNorm0d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, device=device)
        self.alpha = alpha
        self.Vth = Vth
        self.device = device
        self.deactivated = False

    def forward(self, input):
        if (self.deactivated):
            return input
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2])
            # use biased var in train
            var = input.var([0, 2], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():

                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * self.Vth * (input - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None] + self.bias[None, :, None]

        return input