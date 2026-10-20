"""Geometric toy targets; sample-only references, no implemented velocity oracle.

Dataset families follow common FM toy experiments. These are independently
implemented samplers with explicit settings, not exact paper reproductions.
"""
from dataclasses import dataclass
import numpy as np
from synthetic.data import suite


@dataclass
class GeometricTarget:
    kind: str
    noise_std: float = 0.0
    dim: int = 2

    def sample(self, rng, n):
        if self.kind == 'two_moons':
            theta = rng.uniform(0, np.pi, n)
            component = rng.integers(0, 2, n).astype(bool)
            x = np.stack([np.cos(theta), np.sin(theta)], axis=1)
            x[component, 0] = 1-x[component, 0]
            x[component, 1] = 0.5-x[component, 1]
            x -= np.array([0.5, 0.25])
            x += self.noise_std*rng.standard_normal((n, 2))
        elif self.kind == 'checkerboard':
            col = rng.integers(0, 4, n)
            row = col % 2 + 2*rng.integers(0, 2, n)
            x = np.stack([col,row],axis=1) + rng.random((n,2)) - 2
        else:
            raise ValueError(self.kind)
        return x.astype(np.float32)

    def moments(self):
        if self.kind == 'two_moons':
            covariance = np.array([[0.75,0.125-1/np.pi],
                                   [0.125-1/np.pi,0.5625-1/np.pi]])
            return np.zeros(2), covariance + self.noise_std**2*np.eye(2)
        if self.kind == 'checkerboard':
            return np.zeros(2), np.array([[4/3,0.25],[0.25,4/3]])
        raise ValueError(self.kind)

    def to_dict(self):
        return {'kind':self.kind,'noise_std':self.noise_std,'dim':self.dim}


def available_suite():
    return {**suite(), 'two_moons_2d':GeometricTarget('two_moons',noise_std=0.08),
            'checkerboard_2d':GeometricTarget('checkerboard')}
