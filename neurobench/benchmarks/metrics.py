"""
=====================================================================
Project:      NeuroBench
File:         metrics.py
Description:  Python code describing metric for the motor prediction task
Date:         12. May 2023
=====================================================================
Copyright stuff
=====================================================================
"""
import torch


# TODO add computation for ANN
# TODO add computation for bias
def compute_effective_macs(net, spiking_activity):
    """
    Compute the effective macs used in a network
    The number of effective macs consists of three parts
    1) How many effective MACs are needed between the layers
        - for every layer l find the average spiking activity of the incoming spikes o_l
        - multiply o_l with number of neurons from previous layer n_(l-1) and n_l
        - sum_l( o_l * n_(l-1) * n_l)
    2) How many effective MACs are needed within a layer
        - if layer l has explicit recurrence
        - spiking activity o_l times squared number of neurons n_l
        - sum_l( o_l * n_l ** 2 )
    3) How many effective MACs are needed within a neuron
        - how many decaying variables per neuron

    Parameters
    ----------
    net     : nn.Module
        neural network of which to compute the macs from
    spiking_activity : list
        average spiking activity observed during the training per layer

    Returns
    ----------
    macs : int
        effective macs of specific neural network given some data
    """
    # get neurons per layer. This consists of the input neurons x hidden_neurons x output_neurons
    dim = net.get_dimensions()

    # binary of whether a layer comprises an explicit recurrence
    rec_dim = net.get_recurrent_layers()

    # how many decaying variables are used per neuron per layer
    decaying_variables = net.get_decaying_variables()

    # how many bias accumulations are used per neuron per layer
    bias = net.get_bias()

    macs = (spiking_activity * dim[1:] * dim[:-1]).sum() + \
           (spiking_activity * dim[1:] ** 2 * rec_dim).sum() + \
           ((decaying_variables + bias) * dim[1:]).sum()

    return macs


def compute_r2_score(true, pred):
    """
    Compute the average r2 score
    averaged is computed over dimensions of velocity, summed over batches

    Parameters
    ----------
    true    : Tensor
        true velocity
    pred    : Tensor
        predicted velocity

    Returns
    ----------
    r    : float
        r2 score summed over batches
    """
    # compute the sum over time of squares of residuals
    ssres = (true - pred).square().sum(axis=2)

    # compute the total sum over time of squares
    sstot = (true - true.mean()).square().sum(axis=2)

    # compute the coefficient of determination
    r = (1 - ssres / sstot)

    # compute mean over dimensions and sum over batches
    return r.mean(1).sum()


def compute_latency(true, pred, epsilon=1e-2):
    """
    Compute the latency of the prediction
    Latency is defined as the first index when all predictions deviate from the true label by only epsilon

    Parameters
    ----------
    true    : Tensor
        true velocity
    pred    : Tensor
        predicted velocity
    epsilon : float
        allowed error

    Returns
    ----------
    l    : float
        latency of prediction
    """

    # boolean array of indices where the prediction deviates from the true value only by epsilon
    array = ((true - pred).abs() < epsilon).all(0).all(0)

    # base case
    if len(array) == 1:
        return 0

    # exponential search for first nonzero element
    n = 1
    indices = None
    while n < len(array):
        indices = torch.where(array[n - 1:2 * n - 1] != 0)[0]
        if len(indices) > 0:
            return n - 1 + indices[0]
        n = n << 1

    # first nonzero element
    return len(array)
