"""Pruned low-rank K1 maps in a lifted hidden state.

The input/output zero blocks are eliminated exactly at K1. These conditional
linear solves are not asserted to be a single global nonlinear ODE flow.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PrunedLowRankLayer(nn.Module):
    def __init__(self, reference: nn.Module, *, width: int, state_width: int,
                 rank: int, gate_hidden: int, role: str) -> None:
        super().__init__()
        if any(type(value) is not int for value in (width, state_width, rank, gate_hidden)):
            raise ValueError('all low-rank dimensions must be integers')
        if role not in ('lift', 'project') or state_width not in (width, 2*width):
            raise ValueError('use a lift/project K1 map with native or doubled state')
        if not 1 <= rank <= width or not 1 <= gate_hidden <= reference.t_func[0].out_features:
            raise ValueError('rank and controller width exceed the recorded reference')
        if (reference.orders, reference.gate_source, reference.rate_placement,
                reference.solve_return, reference.operator_mode, getattr(reference,'feature_mode',None),
                reference.spectral_cap, reference.forcing_mode, reference.gate_transform,
                reference.clock_curvature, reference.tau, reference.scale, reference.shift) != (
                1, 'input', 'after_projection', 'map', 'multiplicative',
                'independent', None, 'none', 'sigmoid', 1., 10., .8, .2):
            raise ValueError('low-rank maps require the unchanged homogeneous vector K1 reference')
        if (tuple(reference.proj.weight.shape)!=(width,width) or
                reference.t_func[0].in_features!=width or reference.t_func[-1].out_features!=width):
            raise ValueError('declared width must match the vector reference matrices')
        self.width, self.state_width, self.rank, self.gate_hidden = width, state_width, rank, gate_hidden
        self.role = role
        self.input_width = width if role == 'lift' else state_width
        self.output_width = state_width if role == 'lift' else width
        self.tau, self.scale, self.shift = reference.tau, reference.scale, reference.shift
        # SVD is an initializer of the recorded random matrix, never a teacher
        # covariance/rotation. Factor coordinates and global feasible sets differ.
        with torch.no_grad():
            u, singular, vh = torch.linalg.svd(reference.proj.weight.detach().cpu().double(), full_matrices=False)
            root = singular[:rank].sqrt()
            left = (u[:, :rank]*root).to(reference.proj.weight)
            right = (root[:, None]*vh[:rank]).to(reference.proj.weight)
            if role == 'lift' and state_width > width:
                left = torch.cat((left, left.clone()), dim=0)
            if role == 'project' and state_width > width:
                right = F.pad(right, (0, state_width-width))
        self.left, self.right = nn.Parameter(left.clone()), nn.Parameter(right.clone())
        # Constructing new Linear modules must not move any surrounding RNG draw.
        with torch.random.fork_rng(devices=[]):
            self.t_func = nn.Sequential(nn.Linear(self.input_width, gate_hidden), nn.SiLU(),
                                        nn.Linear(gate_hidden, self.output_width))
        self.t_func.to(reference.proj.weight)
        with torch.no_grad():
            self.t_func[0].weight.zero_()
            self.t_func[0].weight[:, :width].copy_(reference.t_func[0].weight[:gate_hidden])
            self.t_func[0].bias.copy_(reference.t_func[0].bias[:gate_hidden])
            self.t_func[-1].weight.zero_()
            self.t_func[-1].bias.zero_()

    def identity(self, x: torch.Tensor) -> torch.Tensor:
        if self.role == 'lift':
            return x if self.state_width == self.width else F.pad(x, (0, self.state_width-self.width))
        return x[..., :self.width]

    def step_size(self, x: torch.Tensor) -> torch.Tensor:
        return self.shift+self.scale*torch.sigmoid(self.t_func(x)/self.tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        carrier = F.linear(F.linear(x, self.right), self.left)
        return self.identity(x)+self.step_size(x)*carrier

    def full_operator(self) -> torch.Tensor:
        rectangular = self.left@self.right
        if self.role == 'lift':return F.pad(rectangular, (0, self.state_width-self.width))
        return F.pad(rectangular, (0, 0, 0, self.state_width-self.width))

    def compressed_operator(self, duration: torch.Tensor) -> torch.Tensor:
        """V D U for the full zero-padded square conditional generator."""
        common = min(self.input_width, self.output_width)
        return self.right[:, :common]@(duration[..., :common, None]*self.left[:common])

    def projection_macs(self) -> int:
        return self.left.numel()+self.right.numel()

    @torch.no_grad()
    def activation_stats(self, x: torch.Tensor) -> dict:
        duration = self.step_size(x)
        carrier = F.linear(F.linear(x, self.right), self.left)
        increment = duration*carrier
        base = self.identity(x)
        rms = lambda z: float(z.square().mean().sqrt())
        denominator = max(rms(x), 1e-12)
        stats = dict(role=self.role, input_width=self.input_width, output_width=self.output_width,
            state_width=self.state_width, rank=self.rank, controller_hidden=self.gate_hidden,
            initialization='constant_clock_recorded_random_SVD_core',
            s_mean=float(duration.mean()), s_std=float(duration.std(unbiased=False)),
            s_min=float(duration.min()), s_max=float(duration.max()),
            s_sample_std=float(duration.std(dim=0, unbiased=False).mean()),
            s_channel_std=float(duration.std(dim=-1, unbiased=False).mean()),
            gate_channels=self.output_width, input_rms=denominator,
            base_rms_over_input=rms(base)/denominator,
            carrier_rms_over_input=rms(carrier)/denominator,
            term1_rms_over_input=rms(increment)/denominator,
            output_rms_over_input=rms(base+increment)/denominator,
            left_factor_norm=float(self.left.norm()), right_factor_norm=float(self.right.norm()))
        stats['extra_output_rms_over_input'] = (rms((base+increment)[..., self.width:])/denominator
            if self.output_width > self.width else 0.)
        stats['extra_input_rms_over_input'] = (rms(x[..., self.width:])/denominator
            if self.input_width > self.width else 0.)
        return stats


class AugmentedODEFFN(nn.Module):
    def __init__(self, reference: nn.Module, *, width: int, state_width: int,
                 rank: int, gate_hidden: int) -> None:
        super().__init__()
        self.ode1 = PrunedLowRankLayer(reference.ode1, width=width, state_width=state_width,
                                     rank=rank, gate_hidden=gate_hidden, role='lift')
        self.act = nn.SiLU()
        self.ode2 = PrunedLowRankLayer(reference.ode2, width=width, state_width=state_width,
                                     rank=rank, gate_hidden=gate_hidden, role='project')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ode2(self.act(self.ode1(x)))
