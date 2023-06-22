import torch

def compute_restricted_outputs(outputs: torch.Tensor, output_size_increment: int):
    """Mask out all but the last output_size_increment number of outputs.

    Args:
        outputs (torch.Tensor): Output tensor from a linear layer.
        output_size_increment (int): Number of outputs to keep.

    Returns:
        torch.Tensor: Masked output tensor.
    """

    num_classes = outputs.size(1)
    restricted_mask = torch.zeros(num_classes)
    restricted_mask[-output_size_increment:] = 1.0
    restricted_mask = restricted_mask.to(outputs.device)

    # Apply the restricted mask to the outputs
    restricted_outputs = outputs * restricted_mask
    
    return restricted_outputs
