"""The original vector clock, frozen or re-evaluated at one internal midpoint.

These solve different auxiliary ODEs. No claim concerns the complete external
FM field's stability, invertibility, or solver accuracy.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class StateClockLayer(nn.Module):
    """Take ownership of a plain vector reference's P/controller, without RNG."""

    def __init__(self, reference: nn.Module, solver: str = 'midpoint'):
        super().__init__()
        if solver not in ('euler', 'frozen', 'midpoint'):
            raise ValueError('solver must be euler, frozen, or midpoint')
        required = dict(feature_mode='independent', gate_source='input',
                        spectral_cap=None, clock_curvature=1., gate_transform='sigmoid',
                        rate_placement='after_projection', solve_return='map',
                        operator_mode='multiplicative')
        if any(not hasattr(reference, name) or getattr(reference, name) != value
               for name, value in required.items()):
            raise ValueError('state clock requires the plain independent input-vector layer')
        if (reference.tau, reference.scale, reference.shift) != (10., .8, .2):
            raise ValueError('state clock retains the original clock constants')
        controller, projection = reference.t_func, reference.proj
        if (not isinstance(projection, nn.Linear) or projection.bias is not None or
                projection.in_features != projection.out_features or
                not isinstance(controller, nn.Sequential) or len(controller) != 3 or
                not isinstance(controller[0], nn.Linear) or not isinstance(controller[1], nn.SiLU) or
                not isinstance(controller[2], nn.Linear) or controller[0].bias is None or controller[2].bias is None or
                controller[0].in_features != projection.in_features or
                controller[0].out_features != controller[2].in_features or
                controller[2].out_features != projection.out_features):
            raise ValueError('expected square bias-free P and a two-linear vector controller')
        parameters = list(reference.named_parameters(remove_duplicate=False))
        if len({id(parameter) for _, parameter in parameters}) != len(parameters):
            raise ValueError('state clock does not accept internally aliased parameters')
        self.proj, self.t_func = projection, controller
        self.tau, self.scale, self.shift = reference.tau, reference.scale, reference.shift
        self.width, self.solver = projection.in_features, solver
        self.projection_calls = 1 if solver == 'euler' else 2
        self.controller_calls = 2 if solver == 'midpoint' else 1
        # This declared paired initialization consumes no random draws. The
        # caller discards the reference; do not register it as another child.
        with torch.no_grad():
            self.t_func[-1].weight.zero_()
            self.t_func[-1].bias.zero_()

    def _validate(self, x):
        if x.ndim < 1 or x.shape[-1] != self.width or not x.is_floating_point():
            raise ValueError('expected floating input with matching final dimension')

    def operator(self):
        return self.proj.weight

    def step_size(self, x):
        self._validate(x)
        return torch.sigmoid(self.t_func(x)/self.tau)*self.scale+self.shift

    def forward(self, x):
        s = self.step_size(x)
        ph = F.linear(x, self.proj.weight)
        first = ph*s
        if self.solver == 'euler':
            return x+first
        pfirst = F.linear(first, self.proj.weight)
        base = x+first+(pfirst*s)/2
        if self.solver == 'frozen':
            return base
        middle = x+first/2
        smid = self.step_size(middle)
        return base+(smid-s)*(ph+pfirst/2)

    def clock_jvp(self, x, direction):
        """Analytic J_s(x)*direction for diagnostics, never used by forward."""
        self._validate(x)
        if direction.shape != x.shape:
            raise ValueError('clock direction must match the input')
        first, _, last = self.t_func
        u = first(x)
        probability = torch.sigmoid(u)
        slope = probability+u*probability*(1-probability)
        logit = last(F.silu(u))
        clock_probability = torch.sigmoid(logit/self.tau)
        tangent = F.linear(slope*F.linear(direction, first.weight), last.weight)
        return (self.scale/self.tau)*clock_probability*(1-clock_probability)*tangent

    @torch.no_grad()
    def activation_stats(self, x):
        """Native-stage statistics plus explicitly labelled counterfactual probes."""
        self._validate(x)
        if not x.numel():
            raise ValueError('statistics require nonempty inputs')
        s = self.step_size(x)
        ph = F.linear(x, self.proj.weight)
        first = s*ph
        middle = x+first/2
        smid = self.step_size(middle)
        pfirst = F.linear(first, self.proj.weight)
        quadratic = (s*pfirst)/2
        correction = (smid-s)*(ph+pfirst/2)
        clock_taylor = .5*ph*self.clock_jvp(x, first)
        cpu = lambda value: value.detach().cpu().double()
        rms = lambda value: float(cpu(value).square().mean().sqrt())
        denominator = max(rms(x), 1e-12)
        stats = dict(scope='internal_state_clock_probes_only', solver=self.solver,
                     projection_calls=self.projection_calls, controller_calls=self.controller_calls,
                     midpoint_clock_is_executed_stage=self.solver == 'midpoint',
                     controller_tau=self.tau, clock_scale=self.scale, clock_shift=self.shift,
                     input_rms=rms(x), gate_change_rms=rms(smid-s),
                     gate_change_abs_max=float((smid-s).abs().max()))
        for name, value in (('initial_clock', s), ('midpoint_clock_probe', smid)):
            tensor = cpu(value)
            stats.update({name+'_mean': float(tensor.mean()), name+'_std': float(tensor.std(unbiased=False)),
                          name+'_min': float(tensor.min()), name+'_max': float(tensor.max())})
        for name, value in (('first_term', first), ('frozen_quadratic_probe', quadratic),
                            ('midpoint_correction_probe', correction), ('clock_taylor_probe', clock_taylor),
                            ('output', self(x))):
            stats[name+'_rms'] = rms(value)
            stats[name+'_rms_over_input'] = rms(value)/denominator
        return stats
