"""Independent dense joint conditioning differentiated by PyTorch."""
from __future__ import annotations

import numpy as np
import torch


def dense_field(value: torch.Tensor, clock: float, rows: np.ndarray,
                basis: np.ndarray, variance: float, head: str) -> torch.Tensor:
    b = torch.tensor(rows, dtype=torch.float64);q = torch.tensor(basis, dtype=torch.float64)
    dim = len(value);a = 1-clock;delta = max(a, .05)
    sigma = variance*torch.eye(dim, dtype=value.dtype)+(.04-variance)*(q@q.T)
    angles = torch.arange(8, dtype=value.dtype)*2*torch.pi/8
    means = 3*torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)@q.T
    covariance = b@(a*a*torch.eye(dim, dtype=value.dtype)+clock*clock*sigma)@b.T
    residual = b@value-clock*(means@b.T)
    solved = torch.linalg.solve(covariance, residual.T).T
    p = torch.softmax(-.5*torch.sum(residual*solved, dim=1), dim=0)
    if head == 'direct':
        component = a/delta*means+(a/delta)*(solved@b@(clock*sigma-a*torch.eye(dim, dtype=value.dtype)))
        return p@component
    component = means+clock*(solved@b@sigma)
    return (p@component-value)/delta


def verify(core) -> dict:
    from field_derivatives import conditional, diagnostics
    rng = np.random.default_rng(0)
    q = np.linalg.qr(rng.standard_normal((10, 2)))[0]
    rows = np.linalg.qr(rng.standard_normal((10, 6)))[0].T
    maximum = finite = full_maximum = commutator = 0.
    for variance in (0., .04):
        for clock in (0., .3, .7, .9):
            x = rng.standard_normal((2, 10))
            for head in ('direct', 'clean', 'shifted'):
                result = conditional(x, clock, rows, q, variance, head)
                function = lambda value: dense_field(value, clock, rows, q, variance, head)
                jacobian = torch.func.vmap(torch.func.jacrev(function))(torch.tensor(x)).numpy()
                discrepancy = float(np.max(np.abs(jacobian-result['jacobian'])))
                maximum = max(maximum, discrepancy);assert discrepancy <= 1e-8
                direction = rng.standard_normal(10);direction /= np.linalg.norm(direction)
                h = 1e-5
                finite_difference = (conditional(x+h*direction, clock, rows, q, variance, head)['field']
                    -conditional(x-h*direction, clock, rows, q, variance, head)['field'])/(2*h)
                error = float(np.max(np.abs(finite_difference-np.einsum('bij,j->bi', result['jacobian'], direction))))
                finite = max(finite, error);assert error <= 1e-6
                projector = rows.T@rows
                predicted_skew = clock/(2*(1-clock)**2)*(result['posterior_covariance']@projector-projector@result['posterior_covariance'])
                observed = (result['clean_jacobian']-result['clean_jacobian'].transpose(0, 2, 1))/2
                commutator = max(commutator, float(np.max(np.abs(predicted_skew-observed))))
                np.testing.assert_allclose(predicted_skew, observed, atol=1e-8, rtol=0)
    q = core.make_basis();x = rng.standard_normal((2, 128))
    for variance in (0., .04):
        for clock in (.3, .7, .9):
            truth = core.oracle(x, np.full((2, 1), clock), q, variance, jacobian=True)['oracle_jacobian']
            for head in ('direct', 'clean', 'shifted'):
                actual = conditional(x, clock, np.eye(128), q, variance, head)['jacobian']
                difference = float(np.max(np.abs(actual-truth)));full_maximum = max(full_maximum, difference)
                assert difference <= 1e-8
    deficient = conditional(x, .7, np.eye(128)[:1], q, 0., 'clean')
    assert np.isfinite(deficient['jacobian']).all()
    diagnostics(rng.standard_normal((2, 128, 128)), deficient['jacobian'], truth, q)
    return dict(status='PASS', automatic_derivative_max=maximum, finite_difference_max=finite,
        full_observation_max=full_maximum, commutator_max=commutator,
        rank_deficient_fixture=True, derivative_cross_term_retained=True)
