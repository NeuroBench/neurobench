"""
Shared helpers for running Neuromorphic Sequential Arena (NSA) baselines in NeuroBench.

NSA (Ma et al., IJCAI 2025, https://arxiv.org/abs/2505.22035) ships its models in the
``neuroseqbench`` package (https://github.com/liyc5929/neuroseqbench). This module does
not copy any NSA code. It only adapts NSA networks to the NeuroBench interfaces:

* NSA networks are time-first: they take ``[T, B, C]`` and return ``[T, B, N]``.
  NeuroBench dataloaders yield batch-first ``[B, T, C]``. ``TimeFirstWrapper`` handles
  the transpose and the mean over time that NSA uses as its readout.
* NSA spiking neurons emit spikes through a ``SpikeGeneration`` module, which is not one
  of NeuroBench's default activation modules. ``wrap_for_neurobench`` registers it so the
  activation hooks (ActivationSparsity) and connection hooks (SynapticOperations) see
  the spikes.
"""

import torch
import torch.nn as nn

from neurobench.models import TorchModel


def nsa_spike_modules():
    """Return the NSA module classes that emit spikes."""
    from neuroseqbench.network.neuron import SpikeGeneration, PMSN_SpikeGeneration

    return [SpikeGeneration, PMSN_SpikeGeneration]


class TimeFirstWrapper(nn.Module):
    """
    Adapts a time-first NSA network to NeuroBench's batch-first convention.

    Args:
        net: NSA network taking ``[T, B, C]`` and returning ``[T, B, N]``.
        readout: ``"mean"`` averages the output over time (NSA's readout for HAR, AL
            and SSL). ``"last"`` keeps the final step (NSA's readout for EEG-MI).

    """

    def __init__(self, net: nn.Module, readout: str = "mean"):
        super().__init__()
        if readout not in ("mean", "last"):
            raise ValueError(f"readout must be 'mean' or 'last', got '{readout}'")
        self.net = net
        self.readout = readout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1).contiguous()  # [B, T, C] -> [T, B, C]
        out = self.net(x)  # [T, B, N]
        if self.readout == "mean":
            return out.mean(0)
        return out[-1]


def wrap_for_neurobench(net: nn.Module, readout: str = "mean") -> TorchModel:
    """
    Wrap an NSA network as a NeuroBench ``TorchModel`` with spike hooks enabled.

    Call this before building the ``Benchmark``, since hooks are registered when the
    Benchmark is constructed.

    """
    model = TorchModel(TimeFirstWrapper(net, readout=readout))
    for module_cls in nsa_spike_modules():
        model.add_activation_module(module_cls)
    return model
