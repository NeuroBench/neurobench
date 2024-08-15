import torch
import snntorch as snn

# static metrics, only require model


def parameter_count(model):
    """
    Number of parameters in the model.

    Args:
        model: A NeuroBenchModel.
    Returns:
        int: Number of parameters.

    """
    return sum(p.numel() for p in model.__net__().parameters())


def footprint(model):
    """
    Memory footprint of the model.

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
    """Sparsity of model connections between layers. This function calculates the sparsity
    based on the number of zero-valued parameters that contribute to connections between
    layers. Only parameters that are involved in connection operations (e.g., matrix
    multiplications) are included in the calculation.
    Supported layers:
    Linear
    Conv1d, Conv2d, Conv3d
    RNN, RNNBase, RNNCell
    LSTM, LSTMBase, LSTMCell
    GRU, GRUBase, GRUCell

    Args:
        model: A NeuroBenchModel.
    Returns:
        float: Connection sparsity, rounded to 3 decimals.
    """

    count_zeros = 0
    count_weights = 0
    for name, param in model.__net__().named_parameters():
        if "weight" in name:
            count_zeros += param.numel() - torch.count_nonzero(param).item()
            count_weights += param.numel()

    # Return the ratio of zeros to weights, rounded to 4 decimals
    return round((count_zeros / count_weights), 4)
