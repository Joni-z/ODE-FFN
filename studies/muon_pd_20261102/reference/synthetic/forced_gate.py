"""Conditional forced-linear responses for an otherwise unchanged SwiGLU gate.

For a fixed gate preactivation a, b=SiLU(a) and c=tanh(eta)*duration(a),
the solution of y'=c*y+b, y(0)=0, at unit internal time is b*phi1(c).
This internal time is not generation time. Bounds on c control response
gain relative to b, not the augmented matrix norm or complete network.
"""
from __future__ import annotations

from numbers import Integral

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def phi1(z: Tensor) -> Tensor:
    """Stable expm1(z)/z with value 1 and derivative 1/2 at zero.

    Degree-eight Horner evaluation is used below |z|=.05. Its value and
    first-derivative truncation errors there are below 5.66e-19 and 1.02e-16.
    These are analytic truncation bounds, not floating-point error bounds.
    The inactive quotient never divides by zero, including during autograd.
    """
    if not z.is_floating_point():
        raise ValueError('phi1 requires a floating-point tensor')
    small = z.abs() < .05
    denominator = torch.where(small, torch.ones_like(z), z)
    quotient = torch.expm1(z) / denominator
    polynomial = torch.full_like(z, 1 / 362880)
    for coefficient in (1/40320, 1/5040, 1/720, 1/120, 1/24, 1/6, .5, 1.):
        polynomial = polynomial * z + coefficient
    return torch.where(small, polynomial, quotient)


class ForcedGateResponse(nn.Module):
    """K2 or exact forced response, with one bounded rate per gate feature."""

    def __init__(self, features: int, solver: str = 'exact',
                 clock: str = 'adaptive') -> None:
        super().__init__()
        if isinstance(features, bool) or not isinstance(features, Integral) or features < 1:
            raise ValueError('features must be a positive integer')
        if solver not in ('k2', 'exact'):
            raise ValueError("solver must be 'k2' or 'exact'")
        if clock not in ('adaptive', 'static'):
            raise ValueError("clock must be 'adaptive' or 'static'")
        self.features, self.solver, self.clock = int(features), solver, clock
        # Zero allocation does not consume RNG or change SwiGLU initialization.
        self.raw_eigenvalues = nn.Parameter(torch.zeros(self.features))

    def _validate_input(self, a: Tensor) -> None:
        if a.ndim < 1 or a.shape[-1] != self.features or not a.is_floating_point():
            raise ValueError('floating input with matching last feature dimension required')

    def eigenvalues(self) -> Tensor:
        return self.raw_eigenvalues.tanh()

    def duration(self, a: Tensor) -> Tensor:
        self._validate_input(a)
        return a.sigmoid() if self.clock == 'adaptive' else torch.full_like(a, .5)

    def _gain_from_rate(self, c: Tensor) -> Tensor:
        return 1 + .5*c if self.solver == 'k2' else phi1(c)

    def gain(self, a: Tensor) -> Tensor:
        return self._gain_from_rate(self.eigenvalues() * self.duration(a))

    def forward(self, a: Tensor) -> Tensor:
        gain = self.gain(a)  # Validate before applying the SiLU kernel.
        return F.silu(a) * gain

    def derivative(self, a: Tensor) -> Tensor:
        """Input derivative including the adaptive duration's contribution."""
        duration = self.duration(a)
        probability = a.sigmoid()
        sigmoid_derivative = probability*(1-probability)
        c = self.eigenvalues()*duration
        gain = self._gain_from_rate(c)
        if self.clock == 'static':
            return (probability+a*sigmoid_derivative)*gain
        response_factor = 1+c if self.solver == 'k2' else c.exp()
        return probability*gain + a*sigmoid_derivative*response_factor

    @torch.no_grad()
    def activation_stats(self, a: Tensor) -> dict[str, float | int | str]:
        """Observed scalar gate statistics; no complete-network bound implied."""
        self._validate_input(a)
        if a.numel() == 0:
            raise ValueError('activation statistics require nonempty inputs')
        duration = self.duration(a)
        eigenvalues = self.eigenvalues()
        conditional_rate = eigenvalues*duration
        gain = self._gain_from_rate(conditional_rate)
        baseline = F.silu(a)
        response = baseline*gain
        derivative = self.derivative(a)
        # MPS does not support float64. Monitoring reductions run on CPU;
        # the gate's forward and derivative remain on their native device.
        rms = lambda x: float(x.detach().cpu().double().square().mean().sqrt())
        baseline_rms = rms(baseline)
        correction_rms = rms(response-baseline)
        return dict(
            scope='observed_gate_response_only', solver=self.solver, clock=self.clock,
            features=self.features, eigenvalue_min=float(eigenvalues.min()),
            eigenvalue_max=float(eigenvalues.max()), eigenvalue_abs_max=float(eigenvalues.abs().max()),
            duration_min=float(duration.min()), duration_max=float(duration.max()),
            duration_mean=float(duration.detach().cpu().double().mean()),
            conditional_rate_min=float(conditional_rate.min()), conditional_rate_max=float(conditional_rate.max()),
            gain_min=float(gain.min()), gain_max=float(gain.max()), gain_mean=float(gain.detach().cpu().double().mean()),
            derivative_min=float(derivative.min()), derivative_max=float(derivative.max()),
            negative_derivative_fraction=float((derivative < 0).detach().cpu().double().mean()),
            baseline_gate_rms=baseline_rms, response_gate_rms=rms(response),
            correction_rms=correction_rms,
            correction_rms_over_baseline=correction_rms/max(baseline_rms, 1e-12),
        )
