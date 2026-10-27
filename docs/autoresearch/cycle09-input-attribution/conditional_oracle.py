"""Exact mixture means after a fixed orthonormal linear observation."""
from __future__ import annotations

import numpy as np


def ring_means() -> np.ndarray:
    angles = np.arange(8)*2*np.pi/8
    return 3*np.stack((np.cos(angles), np.sin(angles)), -1)


def input_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular > 1e-10*max(float(singular[0]), 1.)))
    return right[:rank], singular


def conditional_means(x: np.ndarray, t: np.ndarray, rows: np.ndarray,
                      basis: np.ndarray, variance: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    if np.any((t < 0) | (t >= 1)):
        raise ValueError('This protocol requires 0 <= time < 1')
    np.testing.assert_allclose(rows@rows.T, np.eye(len(rows)), atol=1e-10, rtol=0)
    if len(rows) == 0:
        return np.zeros_like(x), np.zeros_like(x)
    c = rows@basis
    directions, singular, _ = np.linalg.svd(c, full_matrices=False)
    u = x@rows.T
    reduced = u@directions
    component = ring_means()@c.T@directions
    a = 1-t
    d0 = a*a + t*t*variance
    eigenvalues = d0 + t*t*(.04-variance)*singular[None, :]**2
    residual = reduced[:, None, :] - t[:, :, None]*component[None, :, :]
    logits = -.5*np.sum(residual**2/eigenvalues[:, None, :], axis=-1)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    mean = weights@ring_means()
    observed_residual = u - t*(mean@c.T)
    parallel = observed_residual@directions
    perpendicular = observed_residual - parallel@directions.T
    inverse_times_residual = perpendicular/d0 + (parallel/eigenvalues)@directions.T
    # Avoid multiplying a nearly singular normal inverse by C in thin data.
    inverse_c = (parallel/eigenvalues)@(directions.T@c)
    clean = mean@basis.T + t*(variance*(inverse_times_residual@rows)
                               + (.04-variance)*(inverse_c@basis.T))
    noise = a*(inverse_times_residual@rows)
    return clean, noise


def best_field(x: np.ndarray, t: np.ndarray, rows: np.ndarray,
               basis: np.ndarray, variance: float, head: str) -> np.ndarray:
    clean, noise = conditional_means(x, t, rows, basis, variance)
    t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    delta = np.maximum(1-t, .05)
    if head == 'direct':
        return (1-t)/delta*(clean-noise)
    if head not in ('clean', 'shifted'):
        raise ValueError(head)
    return (clean-x)/delta


def components(prediction: np.ndarray, optimum: np.ndarray,
               truth: np.ndarray, basis: np.ndarray) -> np.ndarray:
    information = optimum-truth
    residual = prediction.astype(np.float64)-optimum
    total = prediction.astype(np.float64)-truth
    def values(i, f, v, dimension):
        return np.stack((np.sum(v*v, axis=1), np.sum(i*i, axis=1),
                         np.sum(f*f, axis=1), 2*np.sum(i*f, axis=1)), axis=-1)/dimension
    tangent_i, tangent_f, tangent_v = [value@basis for value in (information, residual, total)]
    full = values(information, residual, total, truth.shape[1])
    tangent = values(tangent_i, tangent_f, tangent_v, basis.shape[1])
    normal = (truth.shape[1]*full-basis.shape[1]*tangent)/(truth.shape[1]-basis.shape[1])
    result = np.stack((full, tangent, normal), axis=1)
    np.testing.assert_allclose(result[:, :, 0], result[:, :, 1:].sum(-1), atol=1e-10, rtol=1e-10)
    return result
