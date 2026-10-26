"""An explicit symmetric sign-code GMM and its exact independent-path FM oracle."""
from __future__ import annotations

import numpy as np


def sign_code(dimension: int = 16) -> np.ndarray:
    if dimension < 1 or dimension & (dimension - 1):
        raise ValueError('Sylvester dimension must be a positive power of two')
    h = np.ones((1, 1), dtype=np.float64)
    while len(h) < dimension:
        h = np.block([[h, h], [h, -h]])
    return np.concatenate([h, -h])


class SignCodeMixture:
    def __init__(self, locations: np.ndarray, rotation: np.ndarray):
        self.locations = np.asarray(locations, dtype=np.float64)
        self.rotation = np.asarray(rotation, dtype=np.float64)
        self.dim = len(self.locations)
        self.signs = sign_code(self.dim)
        self.means = self.signs * self.locations
        self.variance = .04
        np.testing.assert_allclose(self.rotation.T @ self.rotation, np.eye(self.dim), atol=1e-12)

    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray:
        index = rng.integers(len(self.means), size=count)
        latent = self.means[index] + .2 * rng.standard_normal((count, self.dim))
        return (latent @ self.rotation.T).astype(np.float32)

    def oracle(self, x: np.ndarray, time: np.ndarray | float, jacobian: bool = False):
        q = np.asarray(x, dtype=np.float64) @ self.rotation
        t = np.asarray(time, dtype=np.float64)
        if t.ndim == 0:
            t = np.full((len(q), 1), t)
        elif t.ndim == 1:
            t = t[:, None]
        if q.shape != (len(q), self.dim) or t.shape != (len(q), 1) or np.any((t < 0) | (t > 1)):
            raise ValueError('invalid state/time shapes or interpolation time')
        d = (1 - t)**2 + self.variance * t**2
        alpha = (self.variance * t - (1 - t)) / d
        beta = (1 - t) / d
        logits = (t / d * q) @ self.means.T - .5 * t**2 / d * np.sum(self.means**2, axis=1)
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=1, keepdims=True)
        mean = probability @ self.means
        velocity = (alpha * q + beta * mean) @ self.rotation.T
        if not jacobian:
            return velocity
        covariance = (np.einsum('bk,ki,kj->bij', probability, self.means, self.means, optimize=True)
                      - mean[:, :, None] * mean[:, None, :])
        latent_jacobian = (alpha[:, :, None] * np.eye(self.dim)
                           + (t * (1 - t) / d**2)[:, :, None] * covariance)
        physical = np.einsum('ij,bjk,lk->bil', self.rotation, latent_jacobian, self.rotation, optimize=True)
        return velocity, physical, probability

    def to_dict(self) -> dict:
        return dict(kind='symmetric_hadamard_sign_code_gmm', locations=self.locations.tolist(),
                    rotation=self.rotation.tolist(), signs=self.signs.tolist(), variance=self.variance,
                    components=len(self.means), dim=self.dim)


def target_geometry(jacobian: np.ndarray, rotation: np.ndarray) -> dict:
    latent = np.einsum('ji,bjk,kl->bil', rotation, jacobian, rotation, optimize=True)
    diagonal = np.zeros_like(latent)
    idx = np.arange(len(rotation)); diagonal[:, idx, idx] = latent[:, idx, idx]
    denominator = np.sum(jacobian**2, axis=(1, 2))
    off_diagonal = np.sum((latent - diagonal)**2, axis=(1, 2)) / np.maximum(denominator, 1e-30)
    a, b = jacobian[::2], jacobian[1::2]
    commutator = a @ b - b @ a
    commutator_ratio = np.linalg.norm(commutator, axis=(1, 2)) / np.maximum(
        np.linalg.norm(a, axis=(1, 2)) * np.linalg.norm(b, axis=(1, 2)), 1e-30)
    return dict(off_diagonal_energy_fraction=off_diagonal,
                normalized_pair_commutator=commutator_ratio)
