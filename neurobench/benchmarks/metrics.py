import torch

from .utils.metric_utils import check_shape

# TODO: separate out the static and data metrics into different modules

# static metrics, only require model
# TODO: should these be defined by the NeuroBenchModel class or defined here?
def model_size(model):
    """ Memory footprint of the model.

    Args:
        model: A NeuroBenchModel.
    Returns:
        float: Model size in bytes.
    """
    # TODO: should be calculated for nn.Module via parameters/buffers
    return model.size() 

def frequency(model):
    """ Frequency of model forward based on data input window, in Hz.

    Args:
        model: A NeuroBenchModel.
    Returns:
        float: Frequency in Hz.
    """
    # TODO: can also be extracted from the dataset? but prob model to keep consistent
    return model.frequency()

def connection_sparsity(model):
    """ Sparsity of model connections between layers.

    Args:
        model: A NeuroBenchModel.
    Returns:
        float: Connection sparsity.
    """
    # TODO: should be calculated for nn.Module based on number of zeros in Linear and Conv layers.
    return model.connection_sparsity()


# dynamic metrics, require model, model predictions, and labels
def activation_sparsity(model, preds, data):
    """ Sparsity of model activations.
    
    Calculated as the number of zero activations over the total number
    of activations, over all layers, timesteps, samples in data.

    Args:
        model: A NeuroBenchModel.
        preds: A tensor of model predictions.
        data: A tuple of data and labels.
    Returns:
        float: Activation sparsity.
    """
    # TODO: for a spiking model, based on number of spikes over all timesteps over all samples from all layers
    #       Standard FF ANN should be zero (no activation sparsity)
    check_shape(preds, data[1])
    return model.activation_sparsity()

def multiply_accumulates(model, preds, data):
    """ Multiply-accumulates (MACs) of the model forward.

    Args:
        model: A NeuroBenchModel.
        preds: A tensor of model predictions.
        data: A tuple of data and labels.
    Returns:
        float: Multiply-accumulates.
    """
    # TODO: 
    #   Spiking model: number of spike activations * fanout (see snnmetrics)
    #   Recurrent layers: each connection is one MAC
    #   ANN: use PyTorch profiler
    check_shape(preds, data[1])
    macs = 0.0
    return macs

def classification_accuracy(model, preds, data):
    """ Classification accuracy of the model predictions.

    Args:
        model: A NeuroBenchModel.
        preds: A tensor of model predictions.
        data: A tuple of data and labels.
    Returns:
        float: Classification accuracy.
    """
    check_shape(preds, data[1])
    equal = torch.eq(preds, data[1])
    return torch.mean(equal.float()).item()

def MSE(model, preds, data):
    """ Mean squared error of the model predictions.

    Args:
        model: A NeuroBenchModel.
        preds: A tensor of model predictions.
        data: A tuple of data and labels.
    Returns:
        float: Mean squared error.
    """
    check_shape(preds, data[1])
    return torch.mean((preds - data[1])**2).item()

def r2(model, preds, data):
    """ R2 Score of the model predictions.

    Currently implemented for 2D output only.

    Errata: R2 score currently is accumulated by mean for batched eval.
            Currently it is only correct for full-batch evaluation.

    Args:
        model: A NeuroBenchModel.
        preds: A tensor of model predictions.
        data: A tuple of data and labels.
    Returns:
        float: R2 Score.
    """
    X_numerator = torch.sum((data[1][:, 0] - preds[:, 0])**2)
    Y_numerator = torch.sum((data[1][:, 1] - preds[:, 1])**2)
    X_original_label_mean = torch.mean(data[1][:, 0])
    Y_original_label_mean = torch.mean(data[1][:, 1])
    X_denominator = torch.sum((data[1][:, 0] - X_original_label_mean)**2)
    Y_denominator = torch.sum((data[1][:, 1] - Y_original_label_mean)**2)
    X_r2 = 1 - (X_numerator/ X_denominator)
    Y_r2 = 1 - (Y_numerator/ Y_denominator)
    r2 = (X_r2 + Y_r2)/2

    return r2.item()











    