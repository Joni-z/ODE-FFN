"""Diagonal Gaussian mixtures and exact marginal velocity for independent linear FM.

Time convention: x_t = (1-t) * x0 + t * x1; x0 ~ N(0,I), independently of x1.
No component labels are exposed to training. All variances are strictly positive.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class GaussianMixture:
    means: np.ndarray
    variances: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        self.means = np.asarray(self.means, dtype=np.float64)
        self.variances = np.asarray(self.variances, dtype=np.float64)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        if self.means.ndim != 2 or self.variances.shape != self.means.shape:
            raise ValueError('means and variances must have matching [components, dimensions] shapes')
        if self.weights.shape != (len(self.means),):
            raise ValueError('one weight per component is required')
        if not all(np.isfinite(a).all() for a in (self.means, self.variances, self.weights)):
            raise ValueError('parameters must be finite')
        if np.any(self.variances <= 0) or np.any(self.weights <= 0):
            raise ValueError('variances and weights must be positive')
        self.weights = self.weights / self.weights.sum()

    @property
    def dim(self):
        return self.means.shape[1]

    def sample(self, rng, n):
        k = rng.choice(len(self.weights), size=n, p=self.weights)
        return (self.means[k] + np.sqrt(self.variances[k]) * rng.standard_normal((n, self.dim))).astype(np.float32)

    def moments(self):
        mean = self.weights @ self.means
        centered = self.means - mean
        covariance = (centered.T * self.weights) @ centered + np.diag(self.weights @ self.variances)
        return mean, covariance

    def _terms(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError('x must have shape [batch, dimensions]')
        t = np.asarray(t, dtype=np.float64)
        if t.ndim == 0:
            t = np.full((len(x), 1), t)
        elif t.ndim == 1:
            t = t[:, None]
        if t.shape != (len(x), 1) or not np.isfinite(t).all() or np.any((t < 0) | (t > 1)):
            raise ValueError('t must be scalar, [batch], or [batch,1], in [0,1]')
        t = t[:, :, None]
        variance = (1 - t) ** 2 + t ** 2 * self.variances[None]
        residual = x[:, None] - t * self.means[None]
        log_components = (np.log(self.weights)[None]
                          - 0.5 * np.sum(np.log(2 * np.pi * variance) + residual ** 2 / variance, axis=-1))
        log_max = log_components.max(axis=1, keepdims=True)
        probabilities = np.exp(log_components - log_max)
        normalizer = probabilities.sum(axis=1, keepdims=True)
        return t, variance, residual, probabilities / normalizer, (log_max + np.log(normalizer))[:, 0]

    def velocity(self, x, t):
        """Exact E[x1-x0 | x_t=x], not a sampled pair's noisy velocity label."""
        t, variance, residual, responsibilities, _ = self._terms(x, t)
        coefficient = (t * self.variances[None] - (1 - t)) / variance
        component_velocity = self.means[None] + coefficient * residual
        return np.sum(responsibilities[:, :, None] * component_velocity, axis=1)

    def log_density(self, x, t=1.0):
        return self._terms(x, t)[-1]

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in ('means', 'variances', 'weights')}


def suite():
    angles = np.arange(8) * 2 * np.pi / 8
    ring = 3.0 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return {
        'gaussian_iso_2d': GaussianMixture([[1.0, -1.0]], [[1.0, 1.0]], [1]),
        'gaussian_aniso_2d': GaussianMixture([[1.0, -1.0]], [[0.09, 4.0]], [1]),
        'gmm_ring8_2d': GaussianMixture(ring, np.full((8, 2), 0.04), np.ones(8)),
        'gmm_aniso4_2d': GaussianMixture([[-2,-2], [-2,2], [2,-2], [2,2]],
                                       [[0.04,0.64], [0.64,0.04], [0.64,0.04], [0.04,0.64]], np.ones(4)),
        'gaussian_aniso_16d': GaussianMixture(np.zeros((1,16)), np.geomspace(0.04, 4.0, 16)[None], [1]),
    }


def fm_batch(x1, rng):
    """Draw NEW independent source noise and time each call; targets have no labels."""
    x1 = np.asarray(x1, dtype=np.float32)
    x0 = rng.standard_normal(x1.shape).astype(np.float32)
    t = rng.random((len(x1), 1)).astype(np.float32)
    return (1 - t) * x0 + t * x1, t, x1 - x0
