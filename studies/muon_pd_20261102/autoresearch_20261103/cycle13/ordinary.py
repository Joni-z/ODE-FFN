"""Unchanged historical ordinary control; see provenance.json."""
import torch
from torch import nn
from torch.nn import functional as F


def cubic_flow(x: torch.Tensor, duration: torch.Tensor | float) -> torch.Tensor:
    return x * torch.rsqrt(1 + 2 * duration * x.square())



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

