"""Known inner-product potential applied to the complete synthetic field."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from synthetic.models import OriginalODELayer, VelocityNet
from synthetic.recurrent_depth import build as parent_build, copy_surrounding


class ReadoutNet(nn.Module):
    def __init__(self, raw: nn.Module, dim: int, readout: str):
        super().__init__()
        if readout not in ('direct', 'gradient'):
            raise ValueError('unsupported whole-field readout')
        self.raw, self.dim, self.readout = raw, dim, readout
        self.mode = 'native'

    def forward(self, x: torch.Tensor, t: torch.Tensor, *, updates: int = 3,
                step_size: float | None = None) -> torch.Tensor:
        if updates != 3 or step_size is not None:
            raise ValueError('the readout study preserves the three native blocks')
        if self.readout == 'direct':
            return self.raw(x, t)
        keep_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            state = x if x.requires_grad else x.detach().requires_grad_(True)
            raw = self.raw(state, t)
            potential = 0.5 * (state * raw).sum()
            velocity = torch.autograd.grad(potential, state, create_graph=keep_graph)[0]
        return velocity if keep_graph else velocity.detach()

    def ode_layers(self):
        for name, module in self.named_modules():
            if isinstance(module, OriginalODELayer):
                yield name, module


def build(dim: int, family: str, readout: str) -> ReadoutNet:
    if family == 'swiglu384':
        reference = parent_build(dim, 'ode_k2', 'native')
        torch.manual_seed(0)
        raw = VelocityNet(dim, 'swiglu', width=64, depth=3, ffn_hidden=384)
        copy_surrounding(raw, reference.raw)
    else:
        raw = parent_build(dim, family, 'native').raw
    return ReadoutNet(raw, dim, readout)


class MatrixCounter(TorchDispatchMode):
    """Count executed dense matrix MACs, including differentiation operations."""
    def __init__(self):
        super().__init__()
        self.macs = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if name == 'aten.mm.default':
            a, b = args[:2]
            self.macs += a.shape[0] * a.shape[1] * b.shape[1]
        elif name == 'aten.addmm.default':
            a, b = args[1:3]
            self.macs += a.shape[0] * a.shape[1] * b.shape[1]
        elif name == 'aten.bmm.default':
            a, b = args[:2]
            self.macs += a.shape[0] * a.shape[1] * a.shape[2] * b.shape[2]
        elif name in ('aten.mv.default', 'aten.addmv.default'):
            a = args[0] if name == 'aten.mv.default' else args[1]
            self.macs += a.numel()
        return func(*args, **(kwargs or {}))


def compute_budget(model: ReadoutNet) -> dict:
    was_training = model.training
    parameter = next(model.parameters())
    x = torch.linspace(-.4,.7,model.dim,dtype=parameter.dtype,device=parameter.device).reshape(1,-1)
    t = x.new_full((1,1),.4)
    try:
        model.eval()
        with MatrixCounter() as inference:
            with torch.no_grad():
                model(x,t)
        model.train()
        with MatrixCounter() as training:
            prediction = model(x,t)
            loss = prediction.square().mean()
            torch.autograd.grad(loss, tuple(model.parameters()))
        return dict(parameters=sum(p.numel() for p in model.parameters()),
                    dense_forward_macs=int(inference.macs),
                    dense_training_forward_backward_macs=int(training.macs),
                    matrix_count_scope='actual aten mm/addmm/bmm/mv; pointwise work excluded')
    finally:
        model.train(was_training)
