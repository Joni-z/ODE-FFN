"""Controlled hidden-state recurrence around unchanged synthetic FFN blocks."""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from synthetic.models import VelocityNet, OriginalODELayer


FAMILIES = {
    "ode_k2": ("ode_original", {"orders": 2, "gate_hidden": 64}),
    "swiglu192": ("swiglu", {"ffn_hidden": 192}),
    "swiglu128": ("swiglu", {"ffn_hidden": 128}),
}


def copy_surrounding(destination: VelocityNet, source: VelocityNet) -> None:
    with torch.no_grad():
        for getter in ("named_parameters", "named_buffers"):
            target = dict(getattr(destination, getter)())
            reference = dict(getattr(source, getter)())
            keys = {name for name in target if ".ffn." not in name}
            if keys != {name for name in reference if ".ffn." not in name}:
                raise ValueError("surrounding initialization names differ")
            for name in keys:
                if target[name].shape != reference[name].shape:
                    raise ValueError("surrounding initialization shape differs")
                target[name].copy_(reference[name])


class RecurrentDepthNet(nn.Module):
    def __init__(self, raw: VelocityNet, mode: str):
        super().__init__()
        if raw.depth != 3 or mode not in ("native", "cloned", "shared"):
            raise ValueError("this experiment fixes three updates and three layouts")
        if mode == "cloned":
            raw.blocks = nn.ModuleList([copy.deepcopy(raw.blocks[0]) for _ in range(3)])
        elif mode == "shared":
            raw.blocks = nn.ModuleList([raw.blocks[0]])
        self.raw = raw
        self.mode = mode
        self.dim = raw.dim
        self.default_step = 1/math.sqrt(3)

    def forward(self, x: torch.Tensor, t: torch.Tensor, *, return_all: bool = False,
                updates: int = 3, step_size: float | None = None):
        if x.ndim != 2 or x.shape[1] != self.dim or t.shape != (len(x), 1):
            raise ValueError("expected state[batch,dim] and time[batch,1]")
        if type(updates) is not int or updates < 1:
            raise ValueError("updates must be a positive integer")
        if self.mode != "shared" and updates != 3:
            raise ValueError("independent controls use their three trained blocks")
        if step_size is not None and (not math.isfinite(step_size) or step_size <= 0):
            raise ValueError("step size must be positive and finite")
        context = self.raw.time(t)
        hidden = self.raw.input(x)
        predictions = []
        for index in range(updates):
            block = self.raw.blocks[0 if self.mode == "shared" else index]
            if step_size is None or step_size == self.default_step:
                hidden = block(hidden, context)
            else:
                scale, shift = block.condition(context).chunk(2, dim=-1)
                normalized = block.norm(hidden)*(1+scale)+shift
                hidden = hidden+step_size*block.ffn(normalized)
            if return_all or index == updates-1:
                predictions.append(self.raw.output(self.raw.norm(hidden)))
        return predictions if return_all else predictions[-1]

    def ode_layers(self):
        for name, module in self.named_modules():
            if isinstance(module, OriginalODELayer):
                yield name, module


def build(dim: int, family: str, mode: str) -> RecurrentDepthNet:
    if family not in FAMILIES:
        raise ValueError("unsupported FFN family")
    torch.manual_seed(0)
    reference = VelocityNet(dim, "ode_original", width=64, depth=3, gate_hidden=64, orders=2)
    torch.manual_seed(0)
    kind, options = FAMILIES[family]
    raw = VelocityNet(dim, kind, width=64, depth=3, **options)
    copy_surrounding(raw, reference)
    return RecurrentDepthNet(raw, mode)


def training_loss(model: RecurrentDepthNet, x: torch.Tensor, t: torch.Tensor,
                  target: torch.Tensor, supervision: str) -> torch.Tensor:
    if supervision == "final":
        return (model(x, t)-target).square().mean()
    if supervision != "deep":
        raise ValueError("unsupported supervision mode")
    predictions = model(x, t, return_all=True)
    weights = (.25, .25, .5)
    return sum(weight*(prediction-target).square().mean()
               for weight, prediction in zip(weights, predictions))


def compute_budget(model: RecurrentDepthNet, *, updates: int = 3,
                   supervision: str = "final") -> dict:
    counts = {"linear": 0, "operator": 0}
    hooks = []
    def linear(module, args, result):
        counts["linear"] += args[0].numel()*module.out_features
    def operator(module, args, result):
        counts["operator"] += len(args[0])*module.orders*module.proj.weight.numel()
    for module in model.modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear))
        if isinstance(module, OriginalODELayer):
            hooks.append(module.register_forward_hook(operator))
    try:
        with torch.no_grad():
            model(torch.zeros(1, model.dim), torch.full((1, 1), .5),
                  updates=updates, return_all=supervision == "deep")
    finally:
        for hook in hooks:
            hook.remove()
    return {"parameters": sum(parameter.numel() for parameter in model.parameters()),
            "dense_forward_macs": counts["linear"]+counts["operator"],
            "executed_updates": updates, "readout_mode": supervision}
