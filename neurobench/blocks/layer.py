import torch.nn as nn

STATELESS_LAYERS = (
    nn.Linear,
    nn.Conv2d,
    nn.Conv1d,
    nn.Conv3d,
)

RECURRENT_CELLS = (nn.RNNCellBase,)

RECURRENT_LAYERS = (nn.RNNBase,)


SUPPORTED_LAYERS = STATELESS_LAYERS + RECURRENT_LAYERS + RECURRENT_CELLS
