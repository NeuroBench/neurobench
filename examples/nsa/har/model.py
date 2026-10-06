"""
NSA HAR baseline: feedforward (or recurrent) LIF spiking network.

This mirrors the ``FFSNN`` defined in NSA's ``experiments/neuromorphic_sequential_arena/
HAR/main.py`` for the ``--net ffsnn --neuron lif`` configuration. Parameter names
(``fc0``, ``spk0``, ..., ``classifier``) match NSA's, so a checkpoint trained with NSA's
own script loads here with ``load_state_dict`` unchanged.

The neuron itself (``Recurrent_LIF``) and the surrogate gradient come straight from the
``neuroseqbench`` package.
"""

from functools import partial

import torch
import torch.nn as nn

from neuroseqbench.network.neuron import Recurrent_LIF
from neuroseqbench.network.trainer import SurrogateGradient

# Task constants from NSA's HAR setup (WISDM watch gyroscope, 200-step windows).
INPUT_CHANNELS = 3
NUM_CLASSES = 18
TIME_STEPS = 200

# NSA's published HAR hyperparameters for the LIF baselines (HAR/run_all.sh).
LIF_FF_CONFIG = dict(
    hidden_dim=[128, 256, 256], decay=0.9, alpha=0.6, threshold=0.5, recurrent=False
)
LIF_REC_CONFIG = dict(
    hidden_dim=[128, 176, 176], decay=0.1, alpha=0.6, threshold=0.5, recurrent=True
)


class LIFNet(nn.Module):
    """
    Stack of Linear -> LIF layers with a linear readout, time-first.

    Input ``[T, B, C]``, output ``[T, B, num_classes]`` (one logit vector per step).

    """

    def __init__(
        self,
        input_size=INPUT_CHANNELS,
        hidden_dim=(128, 256, 256),
        num_classes=NUM_CLASSES,
        decay=0.9,
        threshold=0.5,
        alpha=0.6,
        recurrent=False,
        time_steps=TIME_STEPS,
    ):
        super().__init__()
        self.num_hidden_layers = len(hidden_dim)
        surro_grad = SurrogateGradient(func_name="triangle", a=alpha)
        neuron = partial(
            Recurrent_LIF,
            decay=decay,
            threshold=threshold,
            time_step=time_steps,
            surro_grad=surro_grad,
            exec_mode="serial",
        )
        in_features = input_size
        for i, width in enumerate(hidden_dim):
            setattr(self, f"fc{i}", nn.Linear(in_features, width))
            # NSA never makes the last hidden layer recurrent for HAR.
            is_last = i == self.num_hidden_layers - 1
            setattr(
                self,
                f"spk{i}",
                neuron(neuron_num=width, recurrent=recurrent and not is_last),
            )
            in_features = width
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        time_steps, batch = x.shape[0], x.shape[1]
        x = x.reshape(time_steps, batch, -1)
        for i in range(self.num_hidden_layers):
            fc = getattr(self, f"fc{i}")
            spk = getattr(self, f"spk{i}")
            # Linear is applied to all steps at once, the neuron then runs over time.
            x = fc(x.reshape(time_steps * batch, -1)).reshape(time_steps, batch, -1)
            x = spk(x)
        x = self.classifier(x.reshape(time_steps * batch, -1))
        return x.reshape(time_steps, batch, -1)


def build_model(recurrent=False, **overrides):
    """Build the NSA LIF baseline with NSA's published HAR config."""
    config = dict(LIF_REC_CONFIG if recurrent else LIF_FF_CONFIG)
    config.update(overrides)
    return LIFNet(**config)


def load_model(path, map_location="cpu"):
    """Load a checkpoint saved by ``train.py`` (or an NSA ``state_dict``)."""
    ckpt = torch.load(path, map_location=map_location)
    if "config" in ckpt:
        model = LIFNet(**ckpt["config"])
        state = ckpt["state_dict"]
    else:  # raw NSA checkpoint: infer the layer widths from the weights
        state = ckpt.get("state_dict", ckpt)
        recurrent = any("recurrent_weight" in k for k in state)
        n_layers = sum(1 for k in state if k.startswith("fc") and k.endswith(".weight"))
        hidden_dim = [state[f"fc{i}.weight"].shape[0] for i in range(n_layers)]
        model = build_model(recurrent=recurrent, hidden_dim=hidden_dim)
    model.load_state_dict(state)
    model.eval()
    return model
