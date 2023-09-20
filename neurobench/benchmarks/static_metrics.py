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
    in supported layers, other layers are not taken into account in the computation:
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

    def get_nr_zeros_weights(module):
        """ Get the number of zeros in a module's weights.

        Args:
            module: A torch.nn.Module.
        Returns:
            int: Number of zeros in the module's weights.
        """
        children = list(module.children())
        
        if len(children) == 0: # it is a leaf
            # print(module)
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Conv3d):
                count_zeros = torch.sum(module.weight == 0)
                count_weights = module.weight.numel()
                return count_zeros, count_weights
            
            elif isinstance(module, torch.nn.RNN) or isinstance(module, torch.nn.RNNBase)or isinstance(module, torch.nn.RNNCell):
                count_zeros = torch.sum(module.weight_hh_l0 == 0) + torch.sum(module.weight_ih_l0 == 0)
                count_weights = module.weight_hh_l0.numel() + module.weight_ih_l0.numel()
                return count_zeros, count_weights
            
            elif isinstance(module, torch.nn.LSTM) or isinstance(module, torch.nn.GRU) or isinstance(module, torch.nn.GRUCell) or isinstance(module, torch.nn.LSTMCell):
                count_zeros = torch.sum(module.weight_hh_l0 == 0) + torch.sum(module.weight_ih_l0 == 0) + torch.sum(module.weight_hh_l0 == 0) + torch.sum(module.weight_ih_l0 == 0)
                count_weights = module.weight_hh_l0.numel() + module.weight_ih_l0.numel() + module.weight_hh_l0.numel() + module.weight_ih_l0.numel()
                return count_zeros, count_weights
            
            else:
                return 0, 0 # it is a neuromorphic neuron layer
        
        else:
            count_zeros = 0
            count_weights = 0
            for child in children:
                child_zeros, child_weights = get_nr_zeros_weights(child)
                count_zeros += child_zeros
                count_weights += child_weights
            return count_zeros, count_weights
    # Pull the layers from the model's network
    layers = model.__net__().children()
    # For each layer, count where the weights are zero
    count_zeros = 0
    count_weights = 0
    for module in layers:
        zeros, weights = get_nr_zeros_weights(module)
        count_zeros += zeros
        count_weights += weights

    # Return the ratio of zeros to weights, rounded to 3 decimals
    return round((count_zeros / count_weights).item(), 3)

