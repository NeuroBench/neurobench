import torch

# static metrics, only require model

def parameter_count(model):
    """ Number of parameters in the model.

    Args:
        model: A NeuroBenchModel.
    Returns:
        int: Number of parameters.
    """
    return sum(p.numel() for p in model.__net__().parameters())

def model_size(model):
    """ Memory footprint of the model.

    Args:
        model: A NeuroBenchModel.
    Returns:
        float: Model size in bytes.
    """
    # Count the number of parameters and multiply by the size of each parameter in bytes
    param_size = 0
    for param in model.__net__().parameters():
        param_size += param.numel() * param.element_size()

    # Count the number of buffers and multiply by the size of each buffer in bytes
    buffer_size = 0
    for buffer in model.__net__().buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    # Return the sum of the parameters and buffers
    return param_size + buffer_size

def connection_sparsity(model):
    """ Sparsity of model connections between layers. Based on number of zeros 
    in Linear and Conv layers.

    Args:
        model: A NeuroBenchModel.
    Returns:
        float: Connection sparsity.
    """
    # Pull the layers from the model's network
    layers = model.__net__().children()

    # For each layer, count where the weights are zero
    count_zeros = 0
    count_weights = 0
    for module in layers:
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            count_zeros += torch.sum(module.weight == 0)
            count_weights += module.weight.numel()

    # Return the ratio of zeros to weights
    return count_zeros / count_weights