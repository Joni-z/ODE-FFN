"""Expanded-feature diagonal ODE maps and their controlled FFN composition.

An adaptive clock is computed from a solve's input and then held fixed. The
exponential is the exact conditional solution of dy/du = diag(lambda) y at
that duration; it is not a solve of dy/du = diag(lambda * s(y)) y with a clock
recomputed along the trajectory. The complete FFN is R psi(E x), a learned
coordinatewise activation MLP. Its middle SiLU need not be invertible, and
scalar-map derivative bounds do not establish complete-network stability.

Relative to the previous dense-P ODEFFN, the expanded features, diagonal maps
and fixed clock change several structural choices. This is not a one-factor
reparameterization of that older model.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch import Tensor, nn


def _positive_size(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _settings(solver: str, clock: str, beta: float) -> float:
    if solver not in ("k1", "k2", "exp"):
        raise ValueError("solver must be 'k1', 'k2' or 'exp'")
    if clock not in ("adaptive", "static"):
        raise ValueError("clock must be 'adaptive' or 'static'")
    if (isinstance(beta, bool) or not isinstance(beta, Real)
            or not math.isfinite(beta) or not 0 < beta <= 1):
        raise ValueError("beta must be finite and satisfy 0 < beta <= 1")
    return float(beta)


class DiagonalFlow(nn.Module):
    """Input-clock scalar maps with one bounded learned eigenvalue per channel.

    The fixed adaptive clock .2 + .8*sigmoid(z) is algebraically the
    professor clock with t_func(z)=10z and tau=10; neither is learned here.
    ``derivative`` includes the derivative of this input-dependent duration.
    ``static`` uses duration .6, including in the derivative calculation.
    """

    def __init__(self, features: int, solver: str = "k2", clock: str = "adaptive",
                 beta: float = 1.) -> None:
        super().__init__()
        self.features = _positive_size(features, "features")
        self.beta = _settings(solver, clock, beta)
        self.solver, self.clock = solver, clock
        # zeros consumes no RNG draws, preserving surrounding initializations.
        self.raw_eigenvalues = nn.Parameter(torch.zeros(self.features))

    def eigenvalues(self) -> Tensor:
        return self.beta * torch.tanh(self.raw_eigenvalues)

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim < 1 or x.shape[-1] != self.features:
            raise ValueError(f"input must have last dimension {self.features}")
        if not x.is_floating_point():
            raise ValueError("input must have a floating-point dtype")

    def duration(self, x: Tensor) -> Tensor:
        self._validate_input(x)
        if self.clock == "static":
            return torch.full_like(x, .6)
        return .2 + .8 * torch.sigmoid(x)

    def _factor(self, y: Tensor) -> Tensor:
        if self.solver == "k1":
            return 1 + y
        if self.solver == "k2":
            return 1 + y + .5 * y.square()
        return torch.exp(y)

    def forward(self, x: Tensor) -> Tensor:
        return x * self._factor(self.eigenvalues() * self.duration(x))

    def derivative(self, x: Tensor) -> Tensor:
        """Return the exact pointwise input derivative, not a frozen-clock one."""
        duration = self.duration(x)
        eigenvalues = self.eigenvalues()
        y = eigenvalues * duration
        factor = self._factor(y)
        if self.clock == "static":
            return factor
        sigmoid = torch.sigmoid(x)
        clock_derivative = .8 * sigmoid * (1 - sigmoid)
        if self.solver == "k1":
            factor_derivative = torch.ones_like(y)
        elif self.solver == "k2":
            factor_derivative = 1 + y
        else:
            factor_derivative = factor
        return factor + x * factor_derivative * eigenvalues * clock_derivative

    @torch.no_grad()
    def activation_stats(self, x: Tensor) -> dict[str, float | str | int]:
        """Observed scalar-map statistics only; no full-network guarantee."""
        if x.numel() == 0:
            raise ValueError("activation statistics require nonempty input")
        duration = self.duration(x)
        eigenvalues = self.eigenvalues()
        factor = self._factor(eigenvalues * duration)
        output = x * factor
        derivative = self.derivative(x)
        # Accumulate monitoring summaries in double without changing the map.
        input_rms = float(x.double().square().mean().sqrt())
        increment_rms = float((output - x).double().square().mean().sqrt())
        return {
            "scope": "observed_coordinatewise_map_only",
            "solver": self.solver,
            "clock": self.clock,
            "features": self.features,
            "beta": self.beta,
            "eigenvalue_min": float(eigenvalues.min()),
            "eigenvalue_max": float(eigenvalues.max()),
            "eigenvalue_abs_max": float(eigenvalues.abs().max()),
            "duration_min": float(duration.min()),
            "duration_max": float(duration.max()),
            "duration_mean": float(duration.double().mean()),
            "duration_std": float(duration.double().std(unbiased=False)),
            "gain_min": float(factor.min()),
            "gain_max": float(factor.max()),
            "derivative_min": float(derivative.min()),
            "derivative_max": float(derivative.max()),
            "negative_derivative_fraction": float((derivative < 0).double().mean()),
            "input_rms": input_rms,
            "output_rms": float(output.double().square().mean().sqrt()),
            "increment_rms": increment_rms,
            "increment_rms_over_input": increment_rms / max(input_rms, 1e-12),
        }


class DiagonalODEFFN(nn.Module):
    """Two coordinatewise maps around SiLU with StandardFFN projection keys.

    Constructing ``net`` exactly as StandardFFN preserves its Linear RNG draws,
    initialization and net.0/net.2 state keys. Only the two eigenvalue vectors
    are added afterward, without consuming RNG. Dense MACs are 2*width*hidden;
    clock and solver pointwise work must be accounted for separately.
    """

    def __init__(self, width: int, hidden: int, solver: str = "k2",
                 clock: str = "adaptive", beta: float = 1.) -> None:
        super().__init__()
        width, hidden = _positive_size(width, "width"), _positive_size(hidden, "hidden")
        beta = _settings(solver, clock, beta)
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, width))
        self.ode1 = DiagonalFlow(hidden, solver, clock, beta)
        self.ode2 = DiagonalFlow(hidden, solver, clock, beta)

    def forward(self, x: Tensor) -> Tensor:
        return self.net[2](self.ode2(self.net[1](self.ode1(self.net[0](x)))))
