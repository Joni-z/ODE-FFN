"""Differentiate the native Heun graph and its empirical target SW2 functional.

The sampler is supplied by the caller; no integration or training rule is copied
or replaced here. The baseline is the actual learned field, never an oracle.
"""
from collections.abc import Callable

import numpy as np
import torch

LAMBDAS = (.125, .25, .5, 1.)


class BlendedField(torch.nn.Module):
    def __init__(self, native: torch.nn.Module, crossed: torch.nn.Module,
                 coefficient: float):
        super().__init__()
        self.native, self.crossed, self.coefficient = native, crossed, coefficient

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.coefficient == 0:
            return self.native(x, t)
        if self.coefficient == 1:
            return self.crossed(x, t)
        value = self.native(x, t)
        return value+self.coefficient*(self.crossed(x, t)-value)


class SensitivityField(torch.nn.Module):
    """Exact real-arithmetic tangent of v_lambda at lambda=0, via JVP."""
    def __init__(self, native: torch.nn.Module, crossed: torch.nn.Module):
        super().__init__()
        self.native, self.crossed = native, crossed
        self.primal_max_error = 0.
        self.primal_qualified = True

    def forward(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x, u = (value.contiguous() for value in state.chunk(2, dim=1))
        direct = self.native(x, t)
        primal, tangent = torch.func.jvp(lambda z: self.native(z, t), (x,), (u,))
        error = torch.abs(primal-direct)
        self.primal_max_error = max(self.primal_max_error, float(error.max()))
        self.primal_qualified &= bool(torch.all(error <= 1e-12*(1+direct.abs())))
        derivative = tangent+self.crossed(x, t)-direct
        return torch.cat((direct, derivative), dim=1)


class RecordedField(torch.nn.Module):
    def __init__(self, field: torch.nn.Module, steps: int):
        super().__init__()
        if steps < 8 or steps % 8:
            raise ValueError('The eight-bin observer requires steps divisible by8')
        self.field, self.steps, self.calls = field, steps, 0
        self.snapshots: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        index, stage = divmod(self.calls, 2)
        assert index < self.steps
        assert bool(torch.all(torch.abs(t-(index+stage)/self.steps) < 1e-13))
        if stage == 0 and index % (self.steps//8) == 0:
            self.snapshots.append(x.detach().clone())
        self.calls += 1
        return self.field(x, t)


def integrate(sampler: Callable, field: torch.nn.Module, source: torch.Tensor,
              steps: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    observer = RecordedField(field, steps)
    endpoint = sampler(observer, source, steps)
    assert observer.calls == 2*steps and len(observer.snapshots) == 8
    observer.snapshots.append(endpoint.clone())
    path = torch.stack(observer.snapshots)
    assert bool(torch.isfinite(path).all())
    return endpoint, path


def squared_sw2(x: np.ndarray, target: np.ndarray, directions: np.ndarray) -> float:
    if x.shape != target.shape:
        raise ValueError('Equal source and target shapes are required')
    residual = np.sort(x@directions.T, axis=0)-np.sort(target@directions.T, axis=0)
    return float(np.mean(residual**2))


def target_derivative(x: np.ndarray, u: np.ndarray, target: np.ndarray,
                      directions: np.ndarray) -> dict:
    """Right derivative; exact ties are ordered by their tangent projection."""
    if x.shape != u.shape or x.shape != target.shape:
        raise ValueError('Endpoint, tangent and target shapes must match')
    projected, tangent = x@directions.T, u@directions.T
    order = np.lexsort((tangent, projected), axis=0)
    sorted_x = np.take_along_axis(projected, order, axis=0)
    sorted_u = np.take_along_axis(tangent, order, axis=0)
    residual = sorted_x-np.sort(target@directions.T, axis=0)
    loss = float(np.mean(residual**2))
    derivative = float(2*np.mean(residual*sorted_u))
    ties = int(np.count_nonzero(np.diff(sorted_x, axis=0) == 0))
    if loss > 0:
        distance_derivative = derivative/(2*np.sqrt(loss))
    else:
        distance_derivative = float(np.sqrt(np.mean(sorted_u**2)))
    checks = []
    for epsilon in (1e-4, 1e-5):
        plus = squared_sw2(x+epsilon*u, target, directions)
        if ties:
            observed = (plus-loss)/epsilon
        else:
            minus = squared_sw2(x-epsilon*u, target, directions)
            observed = (plus-minus)/(2*epsilon)
        error = abs(observed-derivative)
        checks.append(dict(epsilon=epsilon, finite_difference=observed,
            absolute_error=error, qualified=bool(error <= 1e-6+1e-3*abs(derivative))))
    return dict(loss=loss, sw2=float(np.sqrt(loss)), loss_derivative=derivative,
                sw2_derivative=float(distance_derivative), projection_ties=ties,
                finite_differences=checks, qualified=all(r['qualified'] for r in checks))


def prediction_summary(native: np.ndarray, tangent: np.ndarray, endpoints: np.ndarray,
                       target: np.ndarray, directions: np.ndarray) -> dict:
    derivative = target_derivative(native, tangent, target, directions)
    rows = []
    for coefficient, actual in zip(LAMBDAS, endpoints, strict=True):
        displacement = actual-native
        signal = float(np.mean(displacement**2))
        residual = float(np.mean((displacement-coefficient*tangent)**2))
        loss = squared_sw2(actual, target, directions)
        change = loss-derivative['loss']
        predicted = coefficient*derivative['loss_derivative']
        rows.append(dict(coefficient=coefficient, loss=loss, sw2=float(np.sqrt(loss)),
            actual_loss_change=change, predicted_loss_change=predicted,
            prediction_error=predicted-change,
            sign_correct=bool(np.sign(predicted) == np.sign(change)),
            target_fraction=1-(predicted-change)**2/change**2 if change else None,
            endpoint_signal_mse=signal, endpoint_remainder_mse=residual,
            endpoint_fraction=1-residual/signal if signal else None,
            linear_predicted_loss=derivative['loss']+predicted))
    return dict(derivative=derivative, rows=rows)
