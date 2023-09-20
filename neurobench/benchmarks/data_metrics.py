import torch

from .utils.metric_utils import check_shape, make_binary_copy

class AccumulatedMetric:
    """ Abstract class for a metric which must save state between batches.
    """

    def __init__(self):
        """ Initialize metric.
        """
        raise NotImplementedError("Subclasses of AccumulatedMetric should implement __init__")
    
    def __call__(self, model, preds, data):
        """ Process this batch of data.

        Args:
            model: A NeuroBenchModel.
            preds: A torch tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            result: the accumulated metric as of this batch.
        """
        raise NotImplementedError("Subclasses of AccumulatedMetric should implement __call__")

    def compute(self):
        """ Compute the metric score using all accumulated data.

        Returns:
            result: the final accumulated metric.
        """
        raise NotImplementedError("Subclasses of AccumulatedMetric should implement compute")


# dynamic metrics, require model, model predictions, and labels
def activation_sparsity(model, preds, data):
    """ Sparsity of model activations.
    
    Calculated as the number of zero activations over the total number
    def __init__():

    Args:
        model: A NeuroBenchModel.
        preds: A tensor of model predictions.
        data: A tuple of data and labels.
    Returns:
        float: Activation sparsity.
    """
    # TODO: for a spiking model, based on number of spikes over all timesteps over all samples from all layers
    #       Standard FF ANN should be zero (no activation sparsity)
    raise NotImplementedError("Activation sparsity not yet implemented")

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

    binary_model = make_binary_copy(model)
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

def sMAPE(model, preds, data):
    """ Symmetric mean absolute percentage error of the model predictions.

    Args:
        model: A NeuroBenchModel.
        preds: A tensor of model predictions.
        data: A tuple of data and labels.
    Returns:
        float: Symmetric mean absolute percentage error.
    """
    check_shape(preds, data[1])
    smape = 200*torch.mean(torch.abs(preds - data[1])/(torch.abs(preds)+torch.abs(data[1])))
    return torch.nan_to_num(smape, nan=200.0).item()

class r2(AccumulatedMetric):
    """ R2 Score of the model predictions.

    Currently implemented for 2D output only.
    """

    def __init__(self):
        """ Initalize metric state.

        Must hold memory of all labels seen so far.
        """
        self.x_sum_squares = 0.0
        self.y_sum_squares = 0.0
        
        self.x_labels = torch.tensor([])
        self.y_labels = torch.tensor([])

    def __call__(self, model, preds, data):
        """
        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            float: R2 Score.
        """
        check_shape(preds, data[1])
        self.x_sum_squares += torch.sum((data[1][:, 0] - preds[:, 0])**2).item()
        self.y_sum_squares += torch.sum((data[1][:, 1] - preds[:, 1])**2).item()
        self.x_labels = torch.cat((self.x_labels, data[1][:, 0]))
        self.y_labels = torch.cat((self.y_labels, data[1][:, 1]))

        return self.compute()

    def compute(self):
        """ Compute r2 score using accumulated data
        """
        x_denom = self.x_labels.var(correction=0)*len(self.x_labels)
        y_denom = self.y_labels.var(correction=0)*len(self.y_labels)

        x_r2 = 1 - (self.x_sum_squares/ x_denom)
        y_r2 = 1 - (self.y_sum_squares/ y_denom)

        r2 = (x_r2 + y_r2) / 2

        return r2.item()










    