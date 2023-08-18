"""
"""
import torch

from .utils.metric_utils import check_shape

# static metrics, only require model
# TODO: should these be defined by the NeuroBenchModel class or defined here?
def model_size(model):
    '''
    Memory footprint of the model.
    '''
    # TODO: should be calculated for nn.Module via parameters/buffers
    return model.size() 

def frequency(model):
    '''
    Frequency of model forward based on data input window, in Hz.
    '''
    # TODO: can also be extracted from the dataset? but prob model to keep consistent
    return model.frequency()

def connection_sparsity(model):
    '''
    Sparsity of model connections between layers.
    '''
    # TODO: should be calculated for nn.Module based on number of zeros in Linear and Conv layers.
    return model.connection_sparsity()


# dynamic metrics, require model, model predictions, and labels
def activation_sparsity(model, preds, data):
    '''
    Sparsity of model activations.
    '''
    # TODO: for a spiking model, based on number of spikes over all timesteps over all samples from all layers
    #       Standard FF ANN should be zero (no activation sparsity)
    check_shape(preds, data[1])
    return model.activation_sparsity()

def multiply_accumulates(model, preds, data):
    '''
    Multiply-accumulates (MACs) of the model forward.
    '''
    # TODO: 
    #   Spiking model: number of spike activations * fanout (see snnmetrics)
    #   Recurrent layers: each connection is one MAC
    #   ANN: use PyTorch profiler
    check_shape(preds, data[1])
    macs = 0.0
    return macs

def classification_accuracy(model, preds, data):
    '''
    Classification accuracy of the model predictions.
    '''
    check_shape(preds, data[1])
    equal = torch.eq(preds, data[1])
    return torch.mean(equal.float())

def MSE(model, preds, data):
    '''
    Mean squared error of the model predictions.
    '''
    check_shape(preds, data[1])
    return torch.mean((preds - data[1])**2).item()












    