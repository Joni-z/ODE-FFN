"""A fixed two-node approximation of an integrated preactivation flow."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .gated_shear import GatedShearFFN


class NonlinearFeatureFlowFFN(GatedShearFFN):
    """a'=P_C a; u'=v*SiLU(a), with P_C²=0 and v held fixed.

    The path of a is exact. Integrating its SiLU response uses two fixed
    Gauss-Legendre nodes, not an exact terminating matrix exponential.
    C=0 recovers the supplied ordinary SwiGLU projections.
    """

    nodes=(.5-1/(2*math.sqrt(3)),.5+1/(2*math.sqrt(3)))

    def __init__(self,w12: nn.Linear,w3: nn.Linear,*,copy_parameters: bool=True) -> None:
        super().__init__(w12,w3,copy_parameters=copy_parameters)
        width=w3.in_features
        if width<2:raise ValueError('feature flow needs at least two gate channels')
        self.left_width=width//2
        self.transport=nn.Parameter(w12.weight.new_zeros(self.left_width,width-self.left_width))

    def integrated_gate(self,activation: torch.Tensor) -> torch.Tensor:
        left,right=activation.split((self.left_width,activation.shape[-1]-self.left_width),-1)
        displacement=F.linear(right,self.transport)
        low,high=self.nodes
        receiver=.5*(F.silu(left+low*displacement)+F.silu(left+high*displacement))
        return torch.cat((receiver,F.silu(right)),-1)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        activation,value=self.w12(x).chunk(2,-1)
        return self.w3(self.integrated_gate(activation)*value)


class MidpointFeatureFFN(NonlinearFeatureFlowFFN):
    """Ordinary SwiGLU reparameterization: (I+P_C/2) absorbs into Wg,bg."""

    def integrated_gate(self,activation: torch.Tensor) -> torch.Tensor:
        left,right=activation.split((self.left_width,activation.shape[-1]-self.left_width),-1)
        moved=torch.cat((left+.5*F.linear(right,self.transport),right),-1)
        return F.silu(moved)
