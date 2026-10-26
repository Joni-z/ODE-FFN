"""Isolate the centered mean-return term of original second K2 solves."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from synthetic.models import OriginalODEFFN


def second_order_terms(layer: nn.Module, z: torch.Tensor) -> dict[str, torch.Tensor]:
    if (layer.orders != 2 or layer.gate_source != 'input'
            or layer.clock_curvature != 1.0 or layer.spectral_cap is not None
            or layer.rate_placement != 'after_projection' or layer.solve_return != 'map'
            or layer.operator_mode != 'multiplicative' or layer.forcing_mode != 'none'):
        raise ValueError('the diagnosis requires the plain scalar input-clock K2 solve')
    duration = layer.step_size(z)
    if duration.shape[-1] != 1:
        raise ValueError('scalar duration required')
    p = layer.operator()
    u = torch.ones(z.shape[-1], dtype=z.dtype, device=z.device) / math.sqrt(z.shape[-1])
    pz = F.linear(z, p)
    pu = p @ u
    centered_pu = pu - pu.mean()
    returning = 0.5 * duration.square() * (pz @ u).unsqueeze(-1) * centered_pu
    quadratic = 0.5 * duration.square() * F.linear(pz, p)
    return dict(z=z, duration=duration, returning=returning, quadratic=quadratic)


class MeanReturnNet(nn.Module):
    def __init__(self, parent: nn.Module, variant: str = 'native'):
        super().__init__()
        if variant not in ('native', 'drop_uniform', 'drop_return', 'drop_secondorder'):
            raise ValueError('unknown intervention')
        self.raw, self.dim, self.variant = parent.raw, parent.dim, variant
        self.is_ode = all(isinstance(b.ffn, OriginalODEFFN) for b in self.raw.blocks)
        if not self.is_ode and variant not in ('native', 'drop_uniform'):
            raise ValueError('a generic FFN has no second K2 solve')

    def forward(self, x: torch.Tensor, t: torch.Tensor, *, capture: list | None = None) -> torch.Tensor:
        context, hidden = self.raw.time(t), self.raw.input(x)
        for index, block in enumerate(self.raw.blocks):
            scale, shift = block.condition(context).chunk(2, dim=-1)
            normalized = block.norm(hidden) * (1 + scale) + shift
            if self.is_ode:
                ffn = block.ffn
                z = ffn.act(ffn.ode1(normalized))
                branch = ffn.ode2(z)
                if capture is not None or self.variant in ('drop_return', 'drop_secondorder'):
                    terms = second_order_terms(ffn.ode2, z)
                    if capture is not None:
                        capture.append(dict(block=index, **terms))
                    if self.variant == 'drop_return':
                        branch = branch - terms['returning']
                    elif self.variant == 'drop_secondorder':
                        branch = branch - terms['quadratic']
            else:
                branch = block.ffn(normalized)
            if self.variant == 'drop_uniform':
                branch = branch - branch.mean(-1, keepdim=True)
            hidden = hidden + block.residual_scale * branch
        return self.raw.output(self.raw.norm(hidden))
