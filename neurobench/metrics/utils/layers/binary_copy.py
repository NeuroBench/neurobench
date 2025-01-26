from neurobench.blocks.layer import STATELESS_LAYERS, RECURRENT_LAYERS, RECURRENT_CELLS
import copy
import torch


def binarize_tensor(tensor, all_ones=False):
    """
    Binarizes a tensor.

    Non-zero entries become 1. If all_ones is True, all entries become 1.

    """
    tensor = tensor.clone()
    tensor[tensor != 0] = 1
    if all_ones:
        tensor[tensor == 0] = 1
    return tensor


def make_binary_copy(layer, all_ones=False):
    """
    Makes a binary copy of the layer.

    Non-zero entries in the layer's weights and biases are set to 1.
    If all_ones is True, all entries (including zeros) are set to 1.

    Args:
        layer (torch.nn.Module): The layer to be binarized.
        all_ones (bool): If True, all entries (including zeros) are set to 1.

    Returns:
        torch.nn.Module: A binary copy of the input layer.

    """
    layer_copy = copy.deepcopy(layer)

    if isinstance(layer, STATELESS_LAYERS):
        layer_copy.weight.data = binarize_tensor(layer_copy.weight.data, all_ones)
        if layer.bias is not None:
            layer_copy.bias.data = binarize_tensor(layer_copy.bias.data, all_ones)

    elif isinstance(layer, RECURRENT_CELLS):
        attribute_names = ["weight_ih", "weight_hh"]
        if layer.bias:
            attribute_names += ["bias_ih", "bias_hh"]

        for attr in attribute_names:
            with torch.no_grad():
                attr_val = getattr(layer_copy, attr)
                setattr(
                    layer_copy,
                    attr,
                    torch.nn.Parameter(binarize_tensor(attr_val.data, all_ones)),
                )

    return layer_copy
