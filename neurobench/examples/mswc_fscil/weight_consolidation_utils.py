#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This code has been copied from https://github.com/vlomonaco/ar1-pytorch/tree/master
# It is licensed under Creative Commons Attribution 4.0 International License:

# This work is licensed under the Creative Commons Attribution 4.0 International
# License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.

################################################################################
# Copyright (c) 2020. Vincenzo Lomonaco, Gabriele Graffieti, Lorenzo           #
# Pellegrini, Davide Maltoni. All rights reserved.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-04-2020                                                             #
# Authors: Vincenzo Lomonaco, Gabriele Graffieti, Lorenzo Pellegrini, Davide   #
# Maltoni.                                                                     #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

"""
General useful functions for machine learning with Pytorch.
"""

import numpy as np
import torch


def consolidate_weights(model, output, cur_clas, normalize=0):
    """ Mean-shift for the target layer weights"""

    with torch.no_grad():
        globavg = np.average(output.weight.detach()
                             .cpu().numpy()[cur_clas])
        
        if normalize:
            globalnorm = np.std(output.weight.detach()
                                .cpu().numpy()[cur_clas])
        for c in cur_clas:
            w = output.weight.detach().cpu().numpy()[c]

            if c in cur_clas:
                if normalize:
                    new_w = (w - globavg)*normalize/globalnorm
                else:
                    new_w = (w - globavg)
                # if c in model.saved_weights.keys():
                #     wpast_j = np.sqrt(model.past_j[c] / model.cur_j[c])
                #     model.saved_weights[c] = (model.saved_weights[c] * wpast_j
                #      + new_w) / (wpast_j + 1)
                # else:
                model.saved_weights[c] = new_w


def set_consolidate_weights(model, output):
    """ set trained weights """

    with torch.no_grad():
        for c, w in model.saved_weights.items():
            output.weight[c].copy_(
                torch.from_numpy(model.saved_weights[c])
            )


def reset_weights(model, output, cur_clas):
    """ reset weights"""

    with torch.no_grad():
        output.weight.fill_(0.0)
        for c, w in model.saved_weights.items():
            if c in cur_clas:
                output.weight[c].copy_(
                    torch.from_numpy(model.saved_weights[c])
                )


def examples_per_class(train_y, Nmax_classes, shots):
    count = {i:0 for i in range(Nmax_classes)}
    for y in train_y:
        count[int(y)] +=shots

    return count


def eval_below(model, eval_below_layer, only_conv=False):
    # tells whether we want to set model in eval or not
    for name, module in model.named_children():

        if eval_below_layer in name:
            break

        if only_conv:
            if "conv" in name:
                module.eval()
                # print("Freezing parameter " + name)
        else:
            module.eval()
            # print("Freezing parameter " + name)

def freeze_below(model, freeze_below_layer, only_conv=False):
    # tells whether we want to use gradients for a given parameter
    for name, param in model.named_parameters():

        if freeze_below_layer in name:
            break

        elif only_conv:
            if "conv" in name:
                param.requires_grad = False
                # print("Freezing parameter " + name)
        else:
            param.requires_grad = False
            # print("Freezing parameter " + name)
