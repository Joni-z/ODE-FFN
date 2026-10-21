"""Small deterministic checks of the reassessment's identities; no training.

Run from the repository root with the project's PyTorch/NumPy environment.
These checks supplement, rather than replace, the proofs in mathematics.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from synthetic.data import suite

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
torch.set_num_threads(1)


def poly(p, s, k=2):
    total = term = torch.eye(len(p))
    for j in range(1, k + 1):
        term = term @ p * s / j
        total = total + term
    return total


def gate(h):
    return .2 + .8 * torch.sigmoid((h[0] + .3 * h[1] ** 2) / 10)


def main():
    checks = {}
    p = torch.randn(4, 4) / 3
    h = torch.tensor([.4, -.7, .2, .8])
    s = gate(h)
    ds = torch.func.jacrev(gate)(h)
    b = p @ poly(p, s, 1) @ h
    analytic = poly(p, s) + torch.outer(b, ds)
    actual = torch.func.jacrev(lambda x: poly(p, gate(x)) @ x)(h)
    checks['layer_jacobian_max_error'] = (analytic - actual).abs().max().item()

    # The nonlinear ODE f(h)=s(h)Ph has an extra second-order term.
    f = lambda x: gate(x) * (p @ x)
    df_f = torch.func.jacrev(f)(h) @ f(h)
    claimed = s**2 * p @ p @ h + s * torch.dot(ds, p @ h) * (p @ h)
    checks['nonlinear_second_derivative_max_error'] = (df_f - claimed).abs().max().item()
    checks['nonlinear_term_missing_from_frozen_gate'] = (
        .5 * s * torch.dot(ds, p @ h) * (p @ h)).tolist()

    # Exact exponential action still need not be a locally invertible map
    # when its duration depends on the input.
    exact_j = torch.func.jacrev(lambda x: torch.matrix_exp(gate(x) * p) @ x)(h)
    det_formula = torch.exp(s * torch.trace(p)) * (1 + torch.dot(ds, p @ h))
    checks['exact_action_determinant_error'] = abs((torch.det(exact_j) - det_formula).item())

    # Two ODELayers contribute at most rank two beyond frozen-gate chain rule.
    p2 = torch.randn(4, 4) / 3
    y = poly(p, s) @ h
    z = torch.nn.functional.silu(y)
    d = torch.diag(torch.func.jacrev(torch.nn.functional.silu)(y).diag())
    frozen = poly(p2, gate(z)) @ d @ poly(p, s)
    block = lambda x: poly(p2, gate(torch.nn.functional.silu(poly(p, gate(x)) @ x))) @ torch.nn.functional.silu(poly(p, gate(x)) @ x)
    correction = torch.func.jacrev(block)(h) - frozen
    singulars = torch.linalg.svdvals(correction)
    checks['block_correction_singular_values'] = singulars.tolist()

    # Matrix-polynomial parameterization has its own optimization geometry.
    a, direction, g = torch.randn(4, 4) / 3, torch.randn(4, 4), torch.randn(4, 4)
    fun = lambda x: torch.eye(4) + x + .5 * x @ x
    differential = direction + .5 * (a @ direction + direction @ a)
    _, jvp = torch.func.jvp(fun, (a,), (direction,))
    adjoint = g + .5 * (a.T @ g + g @ a.T)
    checks['polynomial_differential_max_error'] = (differential - jvp).abs().max().item()
    checks['polynomial_adjoint_error'] = abs(((differential * g).sum() - (direction * adjoint).sum()).item())
    zero_derivative = direction + .5 * (-torch.eye(4) @ direction + direction @ -torch.eye(4))
    checks['derivative_at_minus_identity_norm'] = torch.linalg.matrix_norm(zero_derivative).item()

    # K2 cannot in general be assigned the invertibility of an exponential.
    a_zero = torch.tensor([[-1., -1.], [1., -1.]])
    checks['k2_singular_example_norm'] = torch.linalg.matrix_norm(torch.eye(2) + a_zero + .5 * a_zero @ a_zero).item()
    # For constant durations the associated operators commute.
    b1, b2 = poly(p, .3), poly(p, .8)
    checks['frozen_operator_commutator_norm'] = torch.linalg.matrix_norm(b1 @ b2 - b2 @ b1).item()

    mixture = suite()['gmm_ring8_2d']
    means = torch.tensor(mixture.means)
    sig2 = .04

    def ring_formula(x, t):
        c = (1-t)**2 + t*t*sig2
        logits = -((x - t*means)**2).sum(-1)/(2*c)
        w = torch.softmax(logits, 0)
        m = w @ means
        centered = means - m
        cov = (centered.T * w) @ centered
        beta = (t*sig2-(1-t))/c
        velocity = beta*x+(1-t)/c*m
        jacobian = beta*torch.eye(2)+t*(1-t)/c**2*cov
        # Posterior covariance of x1 given x_t, including within-mode spread.
        posterior_cov = sig2*(1-t)**2/c*torch.eye(2)+(1-t)**4/c**2*cov
        from_posterior = t/(1-t)**3*posterior_cov-torch.eye(2)/(1-t)
        return velocity, jacobian, from_posterior, w

    rows = []
    max_velocity = max_j = max_cov = 0.
    for t in (.2, .5, .8):
        for name, x in [('center', torch.zeros(2)), ('mode_0', t*means[0]),
                        ('between_0_1', t*(means[0]+means[1])/2)]:
            v, j, jc, weights = ring_formula(x, t)
            numerical = torch.func.jacrev(lambda z: ring_formula(z, t)[0])(x)
            archived = mixture.velocity(x.numpy()[None], t)[0]
            max_velocity = max(max_velocity, float(np.max(np.abs(v.numpy()-archived))))
            max_j = max(max_j, (j-numerical).abs().max().item())
            max_cov = max(max_cov, (j-jc).abs().max().item())
            rows.append({'t':t, 'point':name, 'x':x.tolist(),
                         'jacobian_eigenvalues':torch.linalg.eigvalsh(j).tolist(),
                         'posterior_component_entropy':(-weights*weights.log()).sum().item()})
    checks['ring_velocity_vs_repository_max_error'] = max_velocity
    checks['ring_jacobian_vs_autograd_max_error'] = max_j
    checks['posterior_covariance_identity_max_error'] = max_cov
    checks['ring8_cases'] = rows
    checks['gaussian_half_time_eigenvalues'] = [(.5*l-.5)/(.25+.25*l) for l in (.09,4.)]

    for name, value in checks.items():
        if name.endswith('_error'):
            assert value < 1e-11, (name, value)
    assert singulars[2:].max().item() < 1e-12
    assert checks['k2_singular_example_norm'] == 0
    assert checks['derivative_at_minus_identity_norm'] == 0
    checks['status'] = 'PASS'
    checks['scope'] = 'deterministic mathematical checks; no learned checkpoint evaluation or new training'
    output = Path(__file__).with_name('math-checks.json')
    output.write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
