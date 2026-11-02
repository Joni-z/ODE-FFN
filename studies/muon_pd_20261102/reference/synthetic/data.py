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


@dataclass
class RotatedGaussianMixture:
    """Orthogonal image of a diagonal GMM; Gaussian source remains isotropic."""
    means: np.ndarray
    variances: np.ndarray
    weights: np.ndarray
    rotation: np.ndarray

    def __post_init__(self):
        self.base=GaussianMixture(self.means,self.variances,self.weights)
        self.rotation=np.asarray(self.rotation,dtype=np.float64)
        if self.rotation.shape!=(self.dim,self.dim) or not np.isfinite(self.rotation).all():
            raise ValueError('rotation must be a finite square matrix')
        np.testing.assert_allclose(self.rotation.T@self.rotation,np.eye(self.dim),atol=1e-12,rtol=1e-12)

    @property
    def dim(self):return self.base.dim

    def sample(self,rng,n):
        return (self.base.sample(rng,n).astype(np.float64)@self.rotation.T).astype(np.float32)

    def moments(self):
        mean,cov=self.base.moments()
        return self.rotation@mean,self.rotation@cov@self.rotation.T

    def velocity(self,x,t):
        return self.base.velocity(np.asarray(x)@self.rotation,t)@self.rotation.T

    def log_density(self,x,t=1.):
        return self.base.log_density(np.asarray(x)@self.rotation,t)

    def to_dict(self):
        return {**self.base.to_dict(),'rotation':self.rotation.tolist()}


@dataclass
class RotatedProductMixture:
    """Orthogonal image of independent symmetric two-component mixtures.

    Equivalent to a 2**dim-component GMM, evaluated in O(batch*dim) latent
    work instead of enumerating components. The fixed rotation costs O(dim**2).
    """
    locations: np.ndarray
    variances: np.ndarray
    rotation: np.ndarray

    def __post_init__(self):
        self.locations=np.asarray(self.locations,dtype=np.float64)
        self.variances=np.asarray(self.variances,dtype=np.float64)
        self.rotation=np.asarray(self.rotation,dtype=np.float64)
        if (self.locations.ndim!=1 or self.variances.shape!=self.locations.shape
                or np.any(self.locations<=0) or np.any(self.variances<=0)
                or not all(np.isfinite(a).all() for a in (self.locations,self.variances,self.rotation))):
            raise ValueError('locations and variances must be finite positive vectors')
        if self.rotation.shape!=(self.dim,self.dim):
            raise ValueError('rotation must be square')
        np.testing.assert_allclose(self.rotation.T@self.rotation,np.eye(self.dim),atol=1e-12,rtol=1e-12)

    @property
    def dim(self):return len(self.locations)

    def sample(self,rng,n):
        signs=2*rng.integers(0,2,size=(n,self.dim))-1
        latent=signs*self.locations+np.sqrt(self.variances)*rng.standard_normal((n,self.dim))
        return (latent@self.rotation.T).astype(np.float32)

    def moments(self):
        return np.zeros(self.dim),self.rotation@np.diag(self.locations**2+self.variances)@self.rotation.T

    def _terms(self,x,t):
        x=np.asarray(x,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=self.dim or not np.isfinite(x).all():
            raise ValueError('x must be a finite batch with the declared dimension')
        t=np.asarray(t,dtype=np.float64)
        if t.ndim==0:t=np.full((len(x),1),t)
        elif t.ndim==1:t=t[:,None]
        if t.shape!=(len(x),1) or not np.isfinite(t).all() or np.any((t<0)|(t>1)):
            raise ValueError('invalid interpolation times')
        latent=x@self.rotation
        c=(1-t)**2+t**2*self.variances
        a=(t*self.variances-(1-t))/c
        argument=t*self.locations*latent/c
        posterior_sign=np.tanh(argument)
        velocity=a*latent+(1-t)/c*self.locations*posterior_sign
        derivative=a+t*(1-t)*self.locations**2/c**2*(1-posterior_sign**2)
        logcosh=np.logaddexp(argument,-argument)-np.log(2.)
        log_density=np.sum(-.5*np.log(2*np.pi*c)-(latent**2+t**2*self.locations**2)/(2*c)+logcosh,axis=1)
        probability=(1+posterior_sign)/2
        q=np.clip(probability,np.finfo(float).tiny,1-np.finfo(float).eps)
        entropy=-np.mean(q*np.log(q)+(1-q)*np.log1p(-q),axis=1)/np.log(2.)
        return velocity,derivative,entropy,log_density

    def velocity(self,x,t):return self._terms(x,t)[0]@self.rotation.T

    def log_density(self,x,t=1.):return self._terms(x,t)[-1]

    def jacobian(self,x,t):
        diagonal=self._terms(x,t)[1]
        return np.einsum('ij,bj,kj->bik',self.rotation,diagonal,self.rotation)

    def to_dict(self):
        return {name:getattr(self,name).tolist() for name in ('locations','variances','rotation')}


def suite():
    angles = np.arange(8) * 2 * np.pi / 8
    ring = 3.0 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    rotation,triangular=np.linalg.qr(np.random.default_rng(0).standard_normal((16,16)))
    rotation=rotation*np.sign(np.diag(triangular))[None,:]
    embedded=np.zeros((8,16))
    embedded[:,:2]=ring
    embedded_variances=np.tile(np.r_[np.full(2,.04),np.geomspace(.04,4.,14)],(8,1))
    return {
        'gaussian_iso_2d': GaussianMixture([[1.0, -1.0]], [[1.0, 1.0]], [1]),
        'gaussian_aniso_2d': GaussianMixture([[1.0, -1.0]], [[0.09, 4.0]], [1]),
        'gmm_ring8_2d': GaussianMixture(ring, np.full((8, 2), 0.04), np.ones(8)),
        'gmm_aniso4_2d': GaussianMixture([[-2,-2], [-2,2], [2,-2], [2,2]],
                                       [[0.04,0.64], [0.64,0.04], [0.64,0.04], [0.04,0.64]], np.ones(4)),
        'gaussian_aniso_16d': GaussianMixture(np.zeros((1,16)), np.geomspace(0.04, 4.0, 16)[None], [1]),
        'rotated_gmm_ring8_16d':RotatedGaussianMixture(embedded,embedded_variances,np.ones(8),rotation),
        'rotated_product_bimodal_16d':RotatedProductMixture(np.geomspace(.75,2.5,16),np.full(16,.04),rotation),
    }


def fm_batch(x1, rng):
    """Draw NEW independent source noise and time each call; targets have no labels."""
    x1 = np.asarray(x1, dtype=np.float32)
    x0 = rng.standard_normal(x1.shape).astype(np.float32)
    t = rng.random((len(x1), 1)).astype(np.float32)
    return (1 - t) * x0 + t * x1, t, x1 - x0
