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

from torch.nn.functional import conv1d


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
    spiking_activity : list or int
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

    # how many numbers of multiplies are used per neuron per layer
    bn = net.get_bn()

    macs = (spiking_activity * dim[1:] * dim[:-1]).sum() + \
           (spiking_activity * dim[1:] ** 2 * rec_dim).sum() + \
           ((decaying_variables + bias + bn) * dim[1:]).sum()

    return macs


def compute_r2_score(true, pred, dim=0):
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
    ssres = (true - pred).square().sum(dim=dim)

    # compute the total sum over time of squares
    sstot = (true - true.mean(dim=dim, keepdim=True)).square().sum(dim=dim)

    # compute the coefficient of determination
    r = (1 - ssres / sstot).mean()

    return r


def compute_latency(true, pred, macs, algorithmic_timestep, binning_window, clk_per_op=1, clk_per_memory_access=1,
                reference_clk_freq=1E7):
    """
    Compute the latency of the prediction
    Latency is defined duration between when the first datapoint relevant to the prediction is processed until a
    prediction is made

    Parameters
    ----------
    true    : Tensor
        true velocity
    pred    : Tensor
        predicted velocity
    binning_window : float
        length of binning window (in ms)
    macs : float
        effective MAC operations per algorithmic timestep
    algorithmic_timestep : float
        rate at which network processes input
    clk_per_op : int
        reference clock cycles per MAC operation
    clk_per_memory_access: int
        reference clock cycles per memory access
    reference_clk_freq: int
        reference clock frequency, how many operations per second

    Returns
    ----------
    l    : float
        latency of network
    """

    ops_per_algorithmic_timestep = macs

    if len(true.shape) == 2:
        max_cc = 1
    else:
        max_cc = max_cross_correlation(true, pred)

    ops_per_prediction = ops_per_algorithmic_timestep * max_cc

    # COMMENT: same model with longer algorithmic timestep has a lower ops per second
    ops_per_second = ops_per_prediction / algorithmic_timestep

    # latency_of_operations = (clk_per_op*ops_per_prediction + clk_per_memory_access*memory_access_per_prediction)
    # / reference_clk_freq
    # return binning_window + latency_of_operations
    return ops_per_second


def max_cross_correlation(true, pred):
    """
    Maximize the cross correlation between two nd.arrays

    Parameters
    ----------
    true    : Tensor
        true velocity (batches x nr_features x timestep)
    pred    : Tensor
        predicted velocity (batches x nr_features x timestep)

    Returns
    ----------
    max_cross_correlation    : float
        offset of maximum cross correlation
    """
    # compute cross_correlation over batches x nr_features x timestep
    cross_correlation = conv1d(true, pred, padding=true.shape[2])

    # sum cross correlation over batches and output_features and find argmax
    max_cross_correlation = cross_correlation.sum(dim=(0, 1)).argmax()

    # remove padding and increment by one
    return max(max_cross_correlation - true.shape[2], 0) + 1


def memory_size(net):
    """
    Compute the memory size of a network
    The size of memory consists of three parts
    1) How many floats need to be stored  between the layers
        - multiply  the number of neurons from previous layer n_(l-1) and n_l
        - sum_l( n_(l-1) * n_l)
    2) How many floats are needed within a layer
        - if layer l has explicit recurrence
        - squared number of neurons n_l
        - sum_l( n_l ** 2 )
    3) How many floats are needed within a neuron
        - how many decaying variables per neuron
        - bias
        - batch normalization

    Parameters
    ----------
    net     : nn.Module
        neural network of which to compute the memory size of

    Returns
    ----------
    memory : int
        memory required by neural network
    """
    # get neurons per layer. This consists of the input neurons x hidden_neurons x output_neurons
    dim = net.get_dimensions()

    # binary of whether a layer comprises an explicit recurrence
    rec_dim = net.get_recurrent_layers()

    # how many decaying variables are used per neuron per layer
    decaying_variables = net.get_decaying_variables()

    # how many bias accumulations are used per neuron per layer
    bias = net.get_bias()

    # how many numbers of multiplies are used per neuron per layer
    bn = net.get_bn()

    memory = (dim[1:] * dim[:-1]).sum() + \
           (dim[1:] ** 2 * rec_dim).sum() + \
           ((decaying_variables + bias + bn) * dim[1:]).sum()

    return memory * 32