"""One fixed cubic field and deliberately ordinary nonlinear controls."""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F


def cubic_flow(x: torch.Tensor, duration: torch.Tensor | float) -> torch.Tensor:
    return x * torch.rsqrt(1 + 2 * duration * x.square())


class SplitCore(nn.Module):
    def __init__(self, original: nn.Module, response: str):
        super().__init__()
        if response not in ('cubic', 'silu'):
            raise ValueError('unsupported response')
        self.proj = copy.deepcopy(original.proj)
        self.t_func = copy.deepcopy(original.t_func)
        self.response = response

    def duration(self, x: torch.Tensor) -> torch.Tensor:
        return .2 + .8 * torch.sigmoid(self.t_func(x) / 10.)

    def half(self, x: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        first = .5 * duration * self.proj(x)
        return x + first + .25 * duration * self.proj(first)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        duration = self.duration(x)
        u = self.half(x, duration)
        v = cubic_flow(u, duration) if self.response == 'cubic' else F.silu(u)
        return self.half(v, duration)


class FreeISRU(nn.Module):
    def __init__(self, original: nn.Module):
        super().__init__()
        with torch.no_grad():
            p = original.proj.weight
            initial = torch.eye(len(p), dtype=p.dtype, device=p.device) + .3 * p + .045 * (p @ p)
        self.matrix1 = nn.Parameter(initial.clone())
        self.matrix2 = nn.Parameter(initial.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(cubic_flow(F.linear(x, self.matrix1), .6), self.matrix2)


def explicit_split(module: SplitCore, x: torch.Tensor) -> torch.Tensor:
    """Separate matrix-polynomial expression used only by verification."""
    duration = module.duration(x)
    p = module.proj.weight
    b = (torch.eye(len(p), dtype=x.dtype, device=x.device)
         + .5 * duration[..., None] * p + .125 * duration[..., None].square() * (p @ p))
    u = (b @ x[..., None])[..., 0]
    v = cubic_flow(u, duration) if module.response == 'cubic' else F.silu(u)
    return (b @ v[..., None])[..., 0]


def input_jacobian(module: SplitCore, x: torch.Tensor) -> torch.Tensor:
    """Complete adaptive-duration Jacobian, including its rank-one feedback."""
    state = x.detach().requires_grad_(True)
    duration = module.duration(state)
    grad = torch.autograd.grad(duration.sum(), state)[0]
    p = module.proj.weight
    b = (torch.eye(len(p), dtype=x.dtype, device=x.device)
         + .5 * duration[..., None] * p + .125 * duration[..., None].square() * (p @ p))
    ba = .5 * p + .25 * duration[..., None] * (p @ p)
    u = (b @ state[..., None])[..., 0]
    bu = (ba @ state[..., None])[..., 0]
    if module.response == 'cubic':
        denominator = 1 + 2 * duration * u.square()
        diagonal = denominator.pow(-1.5)
        partial_a = -u.pow(3) * diagonal
        v = cubic_flow(u, duration)
    else:
        sigmoid = torch.sigmoid(u)
        diagonal = sigmoid + u * sigmoid * (1 - sigmoid)
        partial_a = torch.zeros_like(u)
        v = F.silu(u)
    feedback = ((b @ (diagonal * bu + partial_a)[..., None])[..., 0]
                + (ba @ v[..., None])[..., 0])
    return b @ (diagonal[..., :, None] * b) + feedback[..., :, None] * grad[..., None, :]
