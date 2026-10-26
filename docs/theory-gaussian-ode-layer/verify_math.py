#!/usr/bin/env python3
"""Verify the accompanying Gaussian/ODELayer derivation; never train a model.

Usage: python verify_math.py
Requires Python 3.9+, NumPy and SymPy. Writes math-checks.json beside this file.
Symbolic checks verify algebraic identities. Numerical examples supplement, but
do not replace, the proofs and scoped statements in derivation.md.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sympy as sp


CHECKS: list[dict] = []


def exact(name: str, expression) -> None:
    entries = list(expression) if isinstance(expression, sp.MatrixBase) else [expression]
    residuals = [sp.simplify(item) for item in entries]
    if any(item != 0 for item in residuals):
        raise AssertionError(f"{name}: {residuals}")
    CHECKS.append({"name": name, "kind": "symbolic", "entries": len(entries), "passed": True})


def close(name: str, lhs, rhs, tolerance: float = 2e-11) -> None:
    error = float(np.max(np.abs(np.asarray(lhs) - np.asarray(rhs))))
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f"{name}: error={error}, tolerance={tolerance}")
    CHECKS.append({"name": name, "kind": "numeric", "max_abs_error": error,
                   "tolerance": tolerance, "passed": True})


def condition(name: str, passed: bool, **details) -> None:
    if not passed:
        raise AssertionError(name)
    CHECKS.append({"name": name, "kind": "example", "passed": True, **details})


def polynomial(q: sp.MatrixBase, order: int) -> sp.Matrix:
    out = sp.eye(q.rows)
    for index in range(1, order + 1):
        out += q**index / sp.factorial(index)
    return out


def scalar_identities() -> None:
    t, lam, gain, state = sp.symbols("t lambda gain z", real=True)
    b = 1 - t
    c = t**2 * lam + b**2
    a = t * lam / c
    beta = (t * lam - b) / c
    post_var = b**2 * lam / c
    exact("posterior_residual_uncorrelated", t * lam - a * c)
    exact("posterior_covariance", lam - t**2 * lam**2 / c - post_var)
    exact("image_to_velocity", (a - 1) / b - beta)
    exact("posterior_pair_label_variance", lam + 1 - (t * lam - b)**2 / c - lam / c)
    exact("image_risk_decomposition", gain**2 * c - 2 * gain * t * lam + lam
          - ((gain - a)**2 * c + post_var))
    exact("velocity_is_log_variance_derivative", beta - sp.diff(c, t) / (2 * c))
    exact("denoiser_gain_time_derivative", sp.diff(a, t)
          - lam * (1 - (1 + lam) * t**2) / c**2)
    exact("denoiser_posterior_jacobian_identity", a - t * post_var / b**2)
    exact("isotropic_subspace_velocity_factorization", beta + 1 / b - a / b)
    exact("covariance_ode", sp.diff(c, t) - 2 * beta * c)
    log_density_dt = -sp.diff(c, t) / (2 * c) + state**2 * sp.diff(c, t) / (2 * c**2)
    density_div_v = beta - beta * state**2 / c
    exact("gaussian_continuity_equation", log_density_dt + density_div_v)
    exact("positive_variance_velocity_sign_crossing", beta.subs(t, 1 / (1 + lam)))
    exact("normal_image_gain", a.subs(lam, 0))
    exact("normal_velocity", beta.subs(lam, 0) + 1 / b)
    exact("normal_flow_ode", sp.diff(b, t) - beta.subs(lam, 0) * b)
    exact("gaussian_velocity_time_derivative", sp.diff(beta, t)
          - ((lam+1)/c - 2*beta**2))
    exact("normal_external_second_order_term_cancels",
          (sp.diff(beta, t)+beta**2).subs(lam, 0))
    step = sp.symbols("Delta", real=True)
    normal_frozen = 1 - step/b + step**2/(2*b**2)
    exact("normal_frozen_K2_differs_from_exact_flow",
          normal_frozen - (1-t-step)/b - step**2/(2*b**2))
    exact("normalized_denoising_gain", t * a - (1 - b**2 / c))
    exact("at_source_optimal_image_zero", a.subs(t, 0))
    exact("at_source_velocity_minus_identity", beta.subs(t, 0) + 1)

    example_lam = [sp.Integer(4), sp.Integer(1), sp.Integer(0)]
    image_gains = [sp.Rational(8, 5), sp.Integer(1), sp.Integer(0)]
    rates = [sp.Rational(6, 5), sp.Integer(0), sp.Integer(-2)]
    for index, value in enumerate(example_lam):
        exact(f"principal_example_image_{index}", a.subs({t: sp.Rational(1, 2), lam: value})
              - image_gains[index])
        exact(f"principal_example_velocity_{index}", beta.subs({t: sp.Rational(1, 2), lam: value})
              - rates[index])
    exact("thickened_normal_gain", a.subs({t: sp.Rational(1, 2), lam: sp.Rational(1, 100)})
          - sp.Rational(2, 101))


def polynomial_and_gate_identities() -> None:
    q, p, s, c, target = sp.symbols("q p s c a", real=True)
    r = 1 + q + q**2 / 2
    exact("scalar_K2_floor", r - sp.Rational(1, 2) - (q + 1)**2 / 2)
    for value, expected in [(-1, sp.Rational(1, 2)), (-2, 1), (-4, 5)]:
        exact(f"negative_rate_polynomial_{value}", r.subs(q, value) - expected)
    rotation_generator = sp.Matrix([[-1, -1], [1, -1]])
    exact("real_K2_complex_pair_can_annihilate", polynomial(rotation_generator, 2))
    exact("real_K2_matrix_square_identity", polynomial(rotation_generator, 2)
          - ((sp.eye(2) + rotation_generator)**2 + sp.eye(2)) / 2)
    exact("K2_velocity_bridge_quadratic_equation",
          (1+q)**2 - (1+2*((1+q+q**2/2)-1)))
    # This is an exact preimage construction, not a numerical fitting routine.
    j = sp.Matrix([[0, -1], [1, 0]])
    blocks = [-sp.eye(2) + j,
              -sp.eye(2) + sp.sqrt(sp.Rational(1, 2)) * j,
              sp.Matrix([[-1 + sp.sqrt(sp.Rational(11, 5))]])]
    generator = sp.diag(*blocks)
    expected = sp.diag(0, 0, sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(8, 5))
    exact("even_multiplicity_K2_preimage", polynomial(generator, 2) - expected)
    condition("odd_dimensional_negative_square_determinant_obstruction",
              bool((-1)**1 < 0) and bool((-sp.Rational(97, 101))**1 < 0),
              zero_gain_square_rhs=-1, thickened_gain_square_rhs="-97/101")

    x, y = sp.symbols("h1 h2", real=True)
    h = sp.Matrix([x, y])
    gate = sp.Rational(1, 5) + sp.Rational(4, 5) / (1 + sp.exp(-(x - 2*y) / 10))
    grad = sp.Matrix([gate]).jacobian(h)
    mat = sp.Matrix([[1, 2], [-3, 1]])
    for order in (1, 2):
        frozen = polynomial(gate * mat, order)
        field = frozen * h
        correction = mat * polynomial(gate * mat, order - 1) * h * grad
        exact(f"input_dependent_gate_jacobian_K{order}", field.jacobian(h) - frozen - correction)
        exact(f"gate_jacobian_at_origin_K{order}", field.jacobian(h).subs({x: 0, y: 0})
              - frozen.subs({x: 0, y: 0}))

    # Same bounded sigmoid gate form as the professor's layer. A fixed-time
    # exponential is invertible, but this input-dependent stopping map need not be.
    stop = sp.Rational(1, 5) + sp.Rational(4, 5) / (1 + sp.exp(-5 * (x - 1)))
    stop_map = x * sp.exp(-stop)
    exact("state_dependent_exponential_can_have_zero_jacobian", sp.diff(stop_map, x).subs(x, 1))
    exact("state_dependent_exponential_has_strict_local_maximum",
          sp.diff(stop_map, x, 2).subs(x, 1) + sp.exp(-sp.Rational(3, 5)))
    exact("state_dependent_exponential_determinant_formula",
          sp.diff(stop_map, x) - sp.exp(-stop) * (1 - x * sp.diff(stop, x)))

    train_gain = 1 + s * p + s**2 * p**2 / 2
    loss = c * (train_gain - target)**2 / 2
    exact("K2_scalar_loss_derivative", sp.diff(loss, p)
          - c * (train_gain - target) * s * (1 + s * p))
    exact("K2_stationary_parameterization_at_minus_identity", sp.diff(loss, p).subs(p, -1/s))

    eps = sp.symbols("epsilon", real=True)
    mat_p = sp.Matrix([[1, 2], [3, 4]])
    direction = sp.Matrix([[2, -1], [5, 3]])
    perturbed = polynomial(s * (mat_p + eps * direction), 2)
    frechet = s * direction + s**2 * (mat_p * direction + direction * mat_p) / 2
    exact("K2_frechet_derivative", perturbed.diff(eps).subs(eps, 0) - frechet)
    exact("K2_frechet_degeneracy", s * direction
          + s**2 * ((-sp.eye(2)/s) * direction + direction * (-sp.eye(2)/s)) / 2)

    clock = sp.symbols("clock", real=True)
    determinant_polynomial = sp.det(polynomial(clock * mat_p, 2))
    exact("finite_clock_determinant_is_nonzero_polynomial_at_zero",
          determinant_polynomial.subs(clock, 0) - 1)


def matrix_calculations() -> None:
    rng = np.random.default_rng(0)  # deterministic matrix checks, not a training run
    dimension = 6
    basis, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    spectrum = np.array([4., 1., .04, 0., 0., 0.])
    covariance = basis @ np.diag(spectrum) @ basis.T
    identity = np.eye(dimension)

    def objects(time: float):
        b = 1 - time
        cov_z = time**2 * covariance + b**2 * identity
        inverse = np.linalg.inv(cov_z)
        image = time * covariance @ inverse
        velocity = (time * covariance - b * identity) @ inverse
        posterior = covariance - time**2 * covariance @ inverse @ covariance
        return cov_z, inverse, image, velocity, posterior

    for time in [0., .1, .25, .5, .75, .95]:
        label = f"t={time:g}"
        cov_z, inverse, image, velocity, posterior = objects(time)
        b = 1 - time
        close(f"rotated_posterior_residual_cross_covariance_{label}",
              time * covariance - image @ cov_z, np.zeros_like(cov_z))
        close(f"rotated_posterior_covariance_{label}", posterior, b**2 * covariance @ inverse)
        close(f"rotated_image_velocity_bridge_{label}", image - identity, b * velocity)
        close(f"rotated_velocity_covariance_ode_{label}",
              velocity @ cov_z + cov_z @ velocity.T,
              2*time*covariance - 2*b*identity)
        if time:
            close(f"rotated_posterior_jacobian_{label}", image, time * posterior / b**2)
        candidate = rng.normal(size=(dimension, dimension))
        direct_risk = np.trace(candidate @ cov_z @ candidate.T
                               - 2*time*candidate @ covariance + covariance)
        decomposed = np.trace((candidate-image) @ cov_z @ (candidate-image).T + posterior)
        close(f"rotated_population_risk_decomposition_{label}", direct_risk, decomposed)
        close(f"rotated_independent_eigenbasis_image_formula_{label}", image,
              basis @ np.diag(time*spectrum / (time**2*spectrum+b**2)) @ basis.T)
        close(f"rotated_independent_eigenbasis_velocity_formula_{label}", velocity,
              basis @ np.diag((time*spectrum-b) / (time**2*spectrum+b**2)) @ basis.T)

    def flow(start: float, end: float):
        start_c = start**2 * spectrum + (1-start)**2
        end_c = end**2 * spectrum + (1-end)**2
        return basis @ np.diag(np.sqrt(end_c/start_c)) @ basis.T

    start, middle, end = .1, .4, .85
    transport = flow(start, end)
    cov_start = objects(start)[0]
    cov_end = objects(end)[0]
    close("exact_flow_covariance_transport", transport @ cov_start @ transport.T, cov_end)
    close("exact_flow_composition", flow(middle, end) @ flow(start, middle), transport)
    close("exact_flow_identity", flow(start, start), identity)
    delta = 1e-6
    derivative = (flow(start, end+delta)-flow(start, end-delta))/(2*delta)
    close("exact_flow_independent_finite_difference_ode", derivative,
          objects(end)[3] @ transport, tolerance=2e-8)
    exact_limit = basis @ np.diag(np.sqrt(spectrum)) @ basis.T
    close("source_to_endpoint_target_covariance", exact_limit @ exact_limit.T, covariance)
    condition("denoiser_is_not_endpoint_flow", np.linalg.norm(exact_limit) > 1,
              image_prediction_at_source="zero", endpoint_flow_frobenius_norm=float(np.linalg.norm(exact_limit)))

    # The signs cannot be obtained from one fixed generator times a positive clock.
    early_rates = np.array([(0.1*4-.9)/(.1**2*4+.9**2), -1/.9])
    late_rates = np.array([(.5*4-.5)/(.5**2*4+.5**2), -1/.5])
    condition("signal_changes_sign_while_normal_stays_negative",
              early_rates[0] < 0 < late_rates[0] and early_rates[1] < 0 and late_rates[1] < 0,
              early_rates=early_rates.tolist(), late_rates=late_rates.tolist())

    # Constraining eigenvalues alone does not control singular-value growth.
    nonnormal = np.array([[-1., 100.], [0., -1.]])
    q = .5 * nonnormal
    transform = identity[:2, :2] + q + q @ q / 2
    gain = float(np.linalg.svd(transform, compute_uv=False)[0])
    close("nonnormal_K2_same_bounded_eigenvalues", np.linalg.eigvals(transform), [.625, .625])
    condition("nonnormal_K2_can_amplify_with_bounded_eigenvalues", gain > 25,
              operator_norm=gain, generator_eigenvalues=[-1, -1])

    # Check the fixed-clock polynomial pullback/adjoint independently by traces.
    s = .6
    p = rng.normal(size=(3, 3))
    h = rng.normal(size=(3, 3))
    g = rng.normal(size=(3, 3))
    apply_l = lambda z: s*z + .5*s*s*(p@z + z@p)
    apply_adjoint = lambda z: s*z + .5*s*s*(p.T@z + z@p.T)
    close("K2_frechet_frobenius_adjoint", np.sum(apply_l(h)*g), np.sum(h*apply_adjoint(g)))
    close("K2_induced_preconditioner_psd_identity", np.sum(g*apply_l(apply_adjoint(g))),
          np.sum(apply_adjoint(g)**2))


def main() -> None:
    scalar_identities()
    polynomial_and_gate_identities()
    matrix_calculations()
    here = Path(__file__).resolve().parent
    report = {
        "status": "PASS",
        "scope": "Symbolic identities and deterministic matrix examples; no training, no optimizer run, no GPU job.",
        "not_proved_by_checks": [
            "Global representability restrictions are proved with assumptions in derivation.md, not by numerical examples.",
            "No comparison of whole neural-network expressivity or empirical generation quality is asserted.",
            "No ODE-FFN superiority over SwiGLU is established.",
        ],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sympy": sp.__version__,
        "check_count": len(CHECKS),
        "symbolic_check_count": sum(row["kind"] == "symbolic" for row in CHECKS),
        "checks": CHECKS,
        "inputs": {name: hashlib.sha256((here/name).read_bytes()).hexdigest()
                   for name in ["verify_math.py", "derivation.md"]},
    }
    output = here / "math-checks.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"PASS: {len(CHECKS)} checks ({report['symbolic_check_count']} symbolic); no training launched.")
    print(output)


if __name__ == "__main__":
    main()
