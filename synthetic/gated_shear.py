"""An exact conditional shear-flow representation of ordinary SwiGLU.

This introduces no new hypothesis class or quality gain. At fixed FFN input h,
P(h) = [[0, diag(SiLU(Wg h + bg))], [0, 0]] is square-zero. Starting at
(u,v)=(0,Vh+bv), its duration-one endpoint is (g*v,v). The ordinary output
projection is retained. The efficient forward evaluates that exact endpoint.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


class GatedShearFFN(nn.Module):
    """Preserve SwiGLU's projections, parameter order and arithmetic.

    The generator is conditional on the original FFN input, which stays fixed
    during the auxiliary solve. Conditional latent invertibility does not
    imply invertibility or monotonicity of the complete input-output map.
    """

    def __init__(self, w12: nn.Linear, w3: nn.Linear, *, copy_parameters: bool = True) -> None:
        super().__init__()
        if not isinstance(w12, nn.Linear) or not isinstance(w3, nn.Linear):
            raise TypeError('expected the two ordinary SwiGLU Linear projections')
        if w12.out_features != 2 * w3.in_features:
            raise ValueError('packed gate/value width must equal twice the output projection input width')
        self.w12 = copy.deepcopy(w12) if copy_parameters else w12
        self.w3 = copy.deepcopy(w3) if copy_parameters else w3

    @classmethod
    def from_swiglu(cls, module: nn.Module) -> GatedShearFFN:
        """Copy projections from a caller-verified ordinary SwiGLU module."""
        if set(module._modules) != {'w12', 'w3'}:
            raise ValueError('conversion only supports the ordinary two-projection SwiGLU module')
        return cls(module.w12, module.w3)

    @staticmethod
    def generator_apply(gate: torch.Tensor, u: torch.Tensor,
                        v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply P(h) to a lifted state without constructing a dense matrix."""
        if gate.shape != u.shape or gate.shape != v.shape:
            raise ValueError('gate, accumulator and value must have identical shapes')
        return gate * v, torch.zeros_like(v)

    @staticmethod
    def flow(gate: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
             duration: float = 1.) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact e^(duration*P(h)) at fixed conditioning input h."""
        if gate.shape != u.shape or gate.shape != v.shape:
            raise ValueError('gate, accumulator and value must have identical shapes')
        return u + duration * gate * v, v

    @staticmethod
    def zero_accumulator_endpoint(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Duration-one u endpoint; avoid allocating a doubled latent state."""
        return gate * value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.w12(x).chunk(2, dim=-1)
        accumulator = self.zero_accumulator_endpoint(F.silu(gate), value)
        return self.w3(accumulator)
