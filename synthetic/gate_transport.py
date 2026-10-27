"""Exact cube-zero lifted gate transport and an ordinary scaling control."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .gated_shear import GatedShearFFN


class GateTransportFFN(GatedShearFFN):
    """Integrate g'=P_C g, u'=diag(v)g with P_C²=0, then read out u.

    C=0 preserves SwiGLU. The extra half coefficient is the exact time integral,
    not a tuned strength. Conditional nilpotency does not ensure stability of
    the full input-conditioned and normalized network.
    """

    def __init__(self,w12: nn.Linear,w3: nn.Linear,*,copy_parameters: bool=True) -> None:
        super().__init__(w12,w3,copy_parameters=copy_parameters)
        width=w3.in_features
        if width<2:raise ValueError('gate transport needs at least two hidden channels')
        self.left_width=width//2
        self.transport=nn.Parameter(w12.weight.new_zeros(self.left_width,width-self.left_width))

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        activation,value=self.w12(x).chunk(2,dim=-1)
        gate=F.silu(activation)
        left,right=gate.split((self.left_width,gate.shape[-1]-self.left_width),dim=-1)
        moved=torch.cat((left+.5*F.linear(right,self.transport),right),dim=-1)
        return self.w3(moved*value)


class DiagonalGateScaleFFN(GatedShearFFN):
    """Ordinary redundant per-gate scaling; not claimed to be an exact ODE."""

    def __init__(self,w12: nn.Linear,w3: nn.Linear,*,copy_parameters: bool=True) -> None:
        super().__init__(w12,w3,copy_parameters=copy_parameters)
        self.scale_delta=nn.Parameter(w12.weight.new_zeros(w3.in_features))

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        activation,value=self.w12(x).chunk(2,dim=-1)
        gate=F.silu(activation)*(1+.5*self.scale_delta)
        return self.w3(gate*value)
