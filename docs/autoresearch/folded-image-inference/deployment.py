"""Portable inference loader for a frozen, compact image checkpoint."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class CompactSiLU(nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.fc1, self.fc2 = first, second

    def forward(self, x, cond=None, return_aux=False):
        if return_aux:
            raise ValueError('Inference export has no training auxiliary losses.')
        return self.fc2(F.silu(self.fc1(x)))


class CompactSwiGLU(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.w12, self.w3 = original.w12, original.w3

    def forward(self, x, cond=None, return_aux=False):
        if return_aux:
            raise ValueError('Inference-only control.')
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a)*b)


def folded_block(m1, m2):
    d = len(m1)
    first, second = nn.Linear(d,d,bias=False), nn.Linear(d,d,bias=False)
    with torch.no_grad():
        first.weight.copy_(m1)
        second.weight.copy_(m2)
    return CompactSiLU(first,second)


def builder(bundle):
    path = Path(bundle)/'source/image_model.py'
    spec = importlib.util.spec_from_file_location('folded_image_bundle_builder',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def restore(model, state):
    model.load_state_dict(state['model'],strict=True)
    buffers = dict(model.named_buffers())
    assert buffers.keys() == state['runtime_buffers'].keys()
    with torch.no_grad():
        for name, value in state['runtime_buffers'].items():
            buffers[name].copy_(value)
    return model.eval().requires_grad_(False)


def load_export(bundle, device='cpu'):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((bundle/name).read_bytes()).hexdigest() == expected, name
    with torch.random.fork_rng(devices=[]):
        model = builder(bundle).build(bundle/'source/jit','ode_k2')
        for block in model.blocks:
            block.mlp = folded_block(torch.eye(64),torch.eye(64))
    state = torch.load(bundle/'weights.pt',weights_only=True,map_location='cpu')
    restore(model,state)
    return model.to(device)
