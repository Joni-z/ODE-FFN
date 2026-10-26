"""Radial Jacobian-vector features without a hard conservative readout."""
from __future__ import annotations

import torch
from torch import nn

from synthetic.conservative_readout import ReadoutNet,build as parent_build
from synthetic.differentiable_layernorm import replace_normalizations


class DirectionalReadout(ReadoutNet):
    def __init__(self,raw: nn.Module,dim: int):
        replace_normalizations(raw)
        super().__init__(raw,dim,'direct')
        self.readout='directional'

    def forward(self,x: torch.Tensor,t: torch.Tensor,*,updates: int=3,
                step_size: float | None=None) -> torch.Tensor:
        if updates!=3 or step_size is not None:
            raise ValueError('the study preserves the three native blocks')
        if torch.is_inference_mode_enabled():
            raise RuntimeError('forward AD requires no_grad rather than inference_mode')
        value,tangent=torch.func.jvp(lambda state:self.raw(state,t),(x,),(x,))
        return .5*(value+tangent)


def build(dim: int,family: str,readout: str) -> DirectionalReadout:
    if readout!='directional' or family not in ('ode_k2','swiglu128','swiglu192'):
        raise ValueError('unsupported directional study arm')
    return DirectionalReadout(parent_build(dim,family,'direct').raw,dim)
