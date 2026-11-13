"""The project's only model extension: replace the official block FFN."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class FreeISRU(nn.Module):
    """Existing synthetic response; independent square maps, fixed duration0.6.

    The forward and Taylor initialization are copied from the project's
    research-cubic-split-2026-10-26/cubic_core.py. P has the professor layer's
    Linear weight distribution; old experiment RNG positions are not reused.
    """
    def __init__(self, dim):
        super().__init__()
        p = torch.empty(dim, dim)
        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        initial = torch.eye(dim) + .3 * p + .045 * (p @ p)
        self.matrix1 = nn.Parameter(initial.clone())
        self.matrix2 = nn.Parameter(initial.clone())

    def forward(self, x):
        u = F.linear(x, self.matrix1)
        return F.linear(u * torch.rsqrt(1 + 1.2 * u.square()), self.matrix2)


class ISRU_DLayer(nn.Module):
    """Taylor action of exp(P diag(d(h0))); d stays fixed within the series."""

    def __init__(self, dim: int, orders: int = 8) -> None:
        super().__init__()
        if orders < 1:
            raise ValueError('Taylor order must be positive')
        self.orders = orders
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diagonal = torch.rsqrt(1 + 1.2 * x.square())
        term = x
        out = x
        for k in range(1, self.orders + 1):
            term = self.proj(term * diagonal) / k
            out = out + term
        return out


class ISRU_D(nn.Module):
    def __init__(self, dim: int, orders: int = 8, hidden_dim: int = 0) -> None:
        super().__init__()
        if hidden_dim < 0:
            raise ValueError('Internal width must be nonnegative')
        width = hidden_dim or dim
        self.input_proj = nn.Linear(dim, width, bias=False) if width != dim else nn.Identity()
        self.ode1 = ISRU_DLayer(width, orders)
        self.ode2 = ISRU_DLayer(width, orders)
        self.output_proj = nn.Linear(width, dim, bias=False) if width != dim else nn.Identity()
        if width != dim:
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        return self.output_proj(self.ode2(F.silu(self.ode1(x))))


def replace_ffn(model, name, orders=8, hidden_dim=0):
    """Leave SwiGLU exactly native; replace only block.mlp for the candidate."""
    if hidden_dim and name != 'isru_d':
        raise ValueError('Internal ODE width applies only to isru_d')
    if name == 'swiglu':
        return
    if name not in ('isru', 'isru_d'):
        raise ValueError(f'Unknown FFN: {name}')
    # Candidate initialization must not shift the training RNG stream.
    with torch.random.fork_rng(devices=[]):
        for block in model.blocks:
            if block.mlp.ffn_dropout.p != 0:
                raise ValueError('The existing ISRU experiment requires proj_dropout=0')
            dim = block.mlp.w3.out_features
            block.mlp = FreeISRU(dim) if name == 'isru' else ISRU_D(dim, orders, hidden_dim)
