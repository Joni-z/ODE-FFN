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


def replace_ffn(model, name):
    """Leave SwiGLU exactly native; replace only block.mlp for the candidate."""
    if name == 'swiglu':
        return
    if name != 'isru':
        raise ValueError(f'Unknown FFN: {name}')
    # Candidate initialization must not shift the training RNG stream.
    with torch.random.fork_rng(devices=[]):
        for block in model.blocks:
            if block.mlp.ffn_dropout.p != 0:
                raise ValueError('The existing ISRU experiment requires proj_dropout=0')
            block.mlp = FreeISRU(block.mlp.w3.out_features)
