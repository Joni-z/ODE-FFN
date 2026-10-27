"""Spectrum-preserving input frames around the professor's scalar-clock layer.

The frame is chosen once from the layer input, held fixed inside its polynomial,
and differentiated normally. This is not an autonomous state-feedback solver.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from synthetic.models import OriginalODELayer


def rotate(value: torch.Tensor, g: torch.Tensor, *, transpose=False) -> torch.Tensor:
    """H([1,g])H(e1), written as I plus its exact rank-two update.

    The residual form avoids cancelling two sign flips in the backward pass
    when g=0. It is algebraically the same pair of Householder reflectors.
    """
    if transpose:
        g = -g
    squared = g.square().sum(-1, keepdim=True)
    product = (value[..., 1:]*g).sum(-1, keepdim=True)
    first = -2*(squared*value[..., :1]+product)/(1+squared)
    rest = 2*g*(value[..., :1]-product)/(1+squared)
    return value+torch.cat((first, rest), -1)


class ConditionalFrameLayer(OriginalODELayer):
    def __init__(self, source: OriginalODELayer, mode: str = 'state', orders: int = 2):
        nn.Module.__init__(self)
        if mode not in ('state', 'global') or orders not in (1, 2):
            raise ValueError('unsupported conditional frame control')
        required = dict(orders=2, gate_source='input', gate_transform='sigmoid',
            spectral_cap=None, clock_curvature=1., rate_placement='after_projection',
            solve_return='map', operator_mode='multiplicative', forcing_mode='none')
        for key, value in required.items():
            if getattr(source, key) != value:
                raise ValueError(f'conditional frame requires original {key}={value}')
        for key, value in source.__dict__.items():
            if not key.startswith('_'):
                setattr(self, key, value)
        self.t_func, self.proj = source.t_func, source.proj
        self.orders, self.frame_mode = orders, mode
        width, hidden = self.proj.in_features, self.t_func[0].out_features
        self.frame_weight = nn.Parameter(self.proj.weight.new_zeros(width-1, hidden))
        self.frame_bias = nn.Parameter(self.proj.weight.new_zeros(width-1))

    def controls(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.t_func[1](self.t_func[0](x))
        clock = self.duration_from_logit(self.t_func[2](hidden))
        features = hidden if self.frame_mode=='state' else torch.ones_like(hidden)
        g = F.linear(features, self.frame_weight, self.frame_bias)
        direction = torch.cat((torch.ones_like(g[..., :1]), g), -1)
        return clock, g, direction

    def forward(self, x: torch.Tensor, context=None, *, return_duration=False):
        if context is not None:
            raise ValueError('conditional frame retains the original input clock')
        clock, g, _ = self.controls(x)
        term = rotate(x, g, transpose=True)
        output = x
        for index in range(self.orders):
            term = F.linear(term, self.proj.weight)*clock/(index+1)
            output = output+rotate(term, g)
        return (output, clock) if return_duration else output

    @torch.no_grad()
    def activation_stats(self, x: torch.Tensor, context=None) -> dict:
        clock, g, _ = self.controls(x)
        rms = lambda v: float(v.square().mean().sqrt())
        denominator = max(rms(x), 1e-12)
        term = rotate(x, g, transpose=True);output = x
        stats = dict(s_mean=float(clock.mean()), s_std=float(clock.std(unbiased=False)),
            s_min=float(clock.min()), s_max=float(clock.max()), gate_channels=1,
            input_rms=denominator, frame_mode=self.frame_mode, frame_g_rms=rms(g),
            frame_g_sample_std=float(g.std(0, unbiased=False).mean()),
            frame_input_displacement_over_input=rms(term-x)/denominator,
            frame_weight_frobenius=float(self.frame_weight.norm()), frame_bias_norm=float(self.frame_bias.norm()))
        for index in range(self.orders):
            term = F.linear(term, self.proj.weight)*clock/(index+1)
            actual = rotate(term, g)
            stats[f'term{index+1}_rms_over_input'] = rms(actual)/denominator
            output = output+actual
        stats['output_rms_over_input'] = rms(output)/denominator
        return stats
