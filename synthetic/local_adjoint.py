"""Local primal/adjoint FFN and ordinary tied-ridge controls.

The gradient identity is in FFN input coordinates only. Surrounding FiLM,
normalization and residual composition do not preserve a global guarantee.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from synthetic.conservative_readout import build as reference_build
from synthetic.explicit_adjoint import ode_primal, ode_adjoint
from synthetic.models import OriginalODELayer


class LocalAdjointODEFFN(nn.Module):
    def __init__(self, layer: OriginalODELayer, *, include_clock: bool = True):
        super().__init__()
        self.ode1 = copy.deepcopy(layer)
        self.include_clock = include_clock

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        single_vector = h.ndim == 1
        if single_vector:
            h = h.unsqueeze(0)
        encoded, cache = ode_primal(self.ode1, h)
        result = ode_adjoint(cache, F.silu(encoded), include_clock=self.include_clock)
        return result.squeeze(0) if single_vector else result


class TiedSiLUFFN(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.encoder = nn.Linear(width, hidden, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(F.silu(self.encoder(h)), self.encoder.weight.T)


def build(dim: int, setting: str) -> nn.Module:
    if setting in ("swiglu128", "swiglu192"):
        return reference_build(dim, setting, "direct")
    model = reference_build(dim, "ode_k2", "direct")
    if setting == "original":
        return model
    if setting not in ("local_full", "local_frozen_clock", "tied128", "tied192"):
        raise ValueError("unknown local-adjoint setting: " + setting)
    if setting.startswith("tied"):
        torch.manual_seed(0)
    for block in model.raw.blocks:
        if setting.startswith("local"):
            block.ffn = LocalAdjointODEFFN(block.ffn.ode1, include_clock=setting == "local_full")
        else:
            block.ffn = TiedSiLUFFN(block.norm.normalized_shape[0], int(setting[4:]))
    return model
