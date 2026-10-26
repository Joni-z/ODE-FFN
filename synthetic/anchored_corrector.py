"""Same shared FFN, comparing cumulative evolution with input-anchored correction."""
from __future__ import annotations

import math

import torch

from synthetic.recurrent_depth import RecurrentDepthNet, build as build_recurrent


class AnchoredCorrectorNet(RecurrentDepthNet):
    def __init__(self, shared: RecurrentDepthNet, update_rule: str):
        if update_rule not in ("accumulating", "anchored"):
            raise ValueError("unsupported update rule")
        super().__init__(shared.raw, "shared")
        self.update_rule = update_rule

    def hidden_states(self, x: torch.Tensor, t: torch.Tensor, updates: int = 3) -> list[torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != self.dim or t.shape != (len(x), 1):
            raise ValueError("expected state[batch,dim] and time[batch,1]")
        if type(updates) is not int or updates < 1:
            raise ValueError("updates must be a positive integer")
        context = self.raw.time(t)
        anchor = self.raw.input(x)
        hidden = anchor
        states = [hidden]
        block = self.raw.blocks[0]
        for _ in range(updates):
            if self.update_rule == "accumulating":
                hidden = block(hidden, context)
            else:
                scale, shift = block.condition(context).chunk(2, dim=-1)
                normalized = block.norm(hidden)*(1+scale)+shift
                hidden = anchor+self.default_step*block.ffn(normalized)
            states.append(hidden)
        return states

    def forward(self, x: torch.Tensor, t: torch.Tensor, *, return_all: bool = False,
                updates: int = 3, step_size: float | None = None):
        if step_size is not None and (not math.isfinite(step_size) or step_size != self.default_step):
            raise ValueError("the corrector screen fixes its step; iterations do not change its equation")
        states = self.hidden_states(x, t, updates)
        if return_all:
            return [self.raw.output(self.raw.norm(state)) for state in states[1:]]
        return self.raw.output(self.raw.norm(states[-1]))


def build(dim: int, family: str, mode: str) -> AnchoredCorrectorNet:
    return AnchoredCorrectorNet(build_recurrent(dim, family, "shared"), mode)
