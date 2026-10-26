"""Deterministic algebra checks only; no training and no checkpoint changes."""
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
d = 5
P = rng.normal(size=(d, d)) / np.sqrt(d)
h = rng.normal(size=d)
w = rng.normal(size=d)
s0 = 0.6

def gate(x):
    a = 1 / (1 + np.exp(-w @ x / 10))
    return 0.2 + 0.8 * a, 0.08 * a * (1 - a) * w

def terms(x):
    s, _ = gate(x)
    p2 = P @ P
    native = (np.eye(d) + s * P + s * s * p2 / 2) @ x
    linearized = (np.eye(d) + s0 * P + s0 * s0 * p2 / 2
                  + (s - s0) * (P + s0 * p2)) @ x
    return native, linearized

s, u = gate(h)
native, linearized = terms(h)
residual = 0.5 * (s - s0)**2 * (P @ P @ h)
j_residual = 0.5 * (s - s0)**2 * (P @ P) + (s - s0) * np.outer(P @ P @ h, u)
eps = 1e-5
fd = np.column_stack([
    ((terms(h + eps * v)[0] - terms(h + eps * v)[1])
     - (terms(h - eps * v)[0] - terms(h - eps * v)[1])) / (2 * eps)
    for v in np.eye(d)
])

z = 1.3
J = np.array([[0., -1.], [1., 0.]])
T = np.eye(2) + z * J + z*z/2 * (J @ J)
a, b = 0.23, 0.61
T2 = lambda c: np.eye(d) + c * P + c*c/2 * (P @ P)
semigroup_residual = T2(a) @ T2(b) - T2(a+b)
semigroup_formula = a*b*(a+b)/2 * np.linalg.matrix_power(P, 3) + a*a*b*b/4 * np.linalg.matrix_power(P, 4)

results = {
    "description": "Algebra checks, not experiments or a general proof.",
    "seed": 0,
    "centered_expansion_max_abs_error": float(np.max(np.abs(native - linearized - residual))),
    "centered_residual_jacobian_fd_max_abs_error": float(np.max(np.abs(fd - j_residual))),
    "skew_K2_growth_identity_max_abs_error": float(np.max(np.abs(T.T @ T - (1+z**4/4) * np.eye(2)))),
    "semigroup_defect_identity_max_abs_error": float(np.max(np.abs(semigroup_residual - semigroup_formula))),
    "skew_example_z": z,
    "skew_example_T2_singular_values": np.linalg.svd(T, compute_uv=False).tolist(),
}
assert results["centered_expansion_max_abs_error"] < 1e-12
assert results["centered_residual_jacobian_fd_max_abs_error"] < 1e-8
assert results["skew_K2_growth_identity_max_abs_error"] < 1e-12
assert results["semigroup_defect_identity_max_abs_error"] < 1e-12
output = Path(__file__).with_name("new-math-checks.json")
output.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
