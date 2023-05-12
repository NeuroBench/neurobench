'''
This is a SNN taken from the paper
"An Energy-Efficient Spiking Neural Network for Finger Velocity Decoding for Implantable Brain-Machine Interface."
This code is with minor modifications available at "https://github.com/liaoRichard/SNN-for-Finger-Velocity-iBMI"


=====================================================================
Project:      An Energy-Efficient Spiking Neural Network for Finger Velocity Decoding for Implantable Brain-Machine Interface
File:         SNN_baseline_ETH.py
Description:  Python code describing the network architecture
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
Please see the File "LICENCE.md" for the full licensing information.
=====================================================================
'''
import torch
import torch.nn as nn
import numpy as np
from neurobench.models.snn_utils import LIFSpike, LI_no_Spike, tdLayer, tdBatchNorm0d, init_surrogate_gradient


class SNN(nn.Module):
    def __init__(self, input_size, output_size, hyperparams):
        self.hyperparams = hyperparams
        init_surrogate_gradient(hyperparams)
        super(SNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hyperparams['neuron_count']
        self.output_size = output_size

        self.fc1 = tdLayer(nn.Linear(self.input_size,
                                     self.hidden_size,
                                     bias=hyperparams['use_bias'],
                                     device=hyperparams['device']),
                           hyperparam=hyperparams)
        self.dr1 = tdLayer(nn.Dropout(p=hyperparams['dropout']),
                           tdBatchNorm0d(self.hidden_size,
                                         Vth=hyperparams['Vth'],
                                         device=hyperparams['device']),
                           hyperparam=hyperparams)
        self.sp1 = LIFSpike([self.hidden_size], hyperparam=hyperparams)

        self.fc2 = tdLayer(
            nn.Linear(self.hidden_size,
                      self.hidden_size,
                      bias=hyperparams['use_bias'],
                      device=hyperparams['device']),
            hyperparam=hyperparams)
        self.dr2 = tdLayer(nn.Dropout(p=hyperparams['dropout']),
                           tdBatchNorm0d(self.hidden_size, Vth=hyperparams['Vth'],
                                         device=hyperparams['device']),
                           hyperparam=hyperparams)
        self.sp2 = LIFSpike([self.hidden_size], hyperparam=hyperparams)

        self.fc3 = tdLayer(
            nn.Linear(self.hidden_size,
                      self.hidden_size,
                      bias=hyperparams['use_bias'],
                      device=hyperparams['device']),
            hyperparam=hyperparams)
        self.dr3 = tdLayer(nn.Dropout(p=hyperparams['dropout']),
                           tdBatchNorm0d(self.hidden_size,
                                         Vth=hyperparams['Vth'],
                                         device=hyperparams['device']),
                           hyperparam=hyperparams)
        self.sp3 = LIFSpike([self.hidden_size], hyperparam=hyperparams)

        self.out = tdLayer(nn.Linear(self.hidden_size,
                                     self.output_size,
                                     bias=hyperparams['use_bias'],
                                     device=hyperparams['device']),
                           hyperparam=hyperparams)
        self.nospike = LI_no_Spike([self.output_size], hyperparam=hyperparams)
        self.dequant = torch.quantization.DeQuantStub()

    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def forward(self, x):
        input_spiking_activity = torch.mean(x).cpu().detach().numpy()

        x = self.fc1(x)
        x = self.sp1(self.dr1(x))
        # batches x timestep x neurons of hidden layer 1
        spiking_activity1 = torch.mean(x).cpu().detach().numpy()

        x = self.fc2(x)
        x = self.sp2(self.dr2(x))
        # batches x timestep x neurons of hidden layer 2
        spiking_activity2 = torch.mean(x).cpu().detach().numpy()

        x = self.fc3(x)
        x = self.sp3(self.dr3(x))
        # batches x timestep x neurons of hidden layer 3
        spiking_activity3 = torch.mean(x).cpu().detach().numpy()

        x = self.out(x)
        x = self.nospike(x)
        return x, np.array([input_spiking_activity, spiking_activity1, spiking_activity2, spiking_activity3])

    def constrain(self, hyperparam):
        if hyperparam['constrain_method'] == 'eval':
            with torch.no_grad():
                self.sp1.Vth.data = torch.clamp(self.sp1.Vth, min=0)
                self.sp1.tau.data = torch.clamp(self.sp1.tau, min=0, max=1)

                self.sp2.Vth.data = torch.clamp(self.sp2.Vth, min=0)
                self.sp2.tau.data = torch.clamp(self.sp2.tau, min=0, max=1)

                self.sp3.Vth.data = torch.clamp(self.sp3.Vth, min=0)
                self.sp3.tau.data = torch.clamp(self.sp3.tau, min=0, max=1)

                self.nospike.tau.data = torch.clamp(self.nospike.tau, min=0, max=1)

    """
    Added Code
    """

    def get_dimensions(self):
        """
        Get the number of neurons per layer

        Returns
        ----------
        dim    : list of int
            list of neurons per layer
        """
        return np.array([self.input_size, self.hidden_size, self.hidden_size, self.hidden_size, self.output_size])

    def get_recurrent_layers(self):
        """
        Get explicit recurrent layers

        Returns
        ----------
        dim    : list of bool
            boolean for explicit reccurent layers
        """
        return np.array([0, 0, 0, 0])

    def get_decaying_variables(self):
        """
        Get number of decaying variables per neuron per layer

        Returns
        ----------
        dim    : list of int
            number of decaying variables per layer
        """
        return np.array([1, 1, 1, 1])