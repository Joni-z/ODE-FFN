"""Independent posterior moments and target-space Jacobians; no training changes."""
import hashlib
import json
from pathlib import Path
import numpy as np

TIMES = (.1, .3, .5, .7, .9)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def dump(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def posterior(law, x: np.ndarray, t: float, sigma0: float = 1.) -> tuple[np.ndarray, np.ndarray]:
    """E[X|Z=x], Cov[X|Z=x], with Z=t X+(1-t)sigma0 epsilon.

    Compute conditional component moments independently of native velocity code.
    Rotation convention: row-vector world = latent @ R.T.
    """
    if not 0 < t < 1 or sigma0 <= 0:
        raise ValueError('Posterior identity uses interior time and positive noise scale')
    x = np.asarray(x, dtype=np.float64)
    rotation = np.asarray(getattr(law, 'rotation', np.eye(law.dim)))
    z = x@rotation
    beta2 = ((1-t)*sigma0)**2
    if hasattr(law, 'locations'):
        var, mu = law.variances, law.locations
        c = beta2+t*t*var
        sign = np.tanh(t*mu*z/c)
        mean = t*var*z/c+beta2*mu*sign/c
        diagonal = var*beta2/c+(beta2*mu/c)**2*(1-sign*sign)
        covariance = rotation[None]*diagonal[:, None, :]
        covariance = covariance@rotation.T
    else:
        raw = getattr(law, 'base', law)
        c = beta2+t*t*raw.variances
        residual = z[:, None]-t*raw.means
        logits = np.log(raw.weights)-.5*np.sum(np.log(2*np.pi*c)+residual**2/c, axis=2)
        weight = np.exp(logits-logits.max(axis=1, keepdims=True))
        weight /= weight.sum(axis=1, keepdims=True)
        component_mean = raw.means+t*raw.variances*residual/c
        mean = np.einsum('nk,nkd->nd', weight, component_mean)
        centered = component_mean-mean[:, None]
        covariance = np.einsum('nk,nki,nkj->nij', weight, centered, centered)
        diagonal = np.einsum('nk,kd->nd', weight, raw.variances*beta2/c)
        index = np.arange(law.dim)
        covariance[:, index, index] += diagonal
        covariance = rotation@covariance@rotation.T
    return mean@rotation.T, covariance


def target(law, x: np.ndarray, t: float, sigma0: float = 1.) -> dict:
    mean, covariance = posterior(law, x, t, sigma0)
    jm = t/(((1-t)*sigma0)**2)*covariance
    return dict(mean=mean, covariance=covariance, image_jacobian=jm,
                velocity=(mean-x)/(1-t),
                velocity_jacobian=(jm-np.eye(law.dim))/(1-t))


def metrics(prediction: np.ndarray, jacobian: np.ndarray, oracle: np.ndarray,
            oracle_jacobian: np.ndarray, t: float) -> dict:
    """Equal-state/time metrics; Frobenius energy / d = mean unit-direction energy."""
    dim = prediction.shape[1]
    jm = np.eye(dim)+(1-t)*jacobian
    sym = (jm+jm.transpose(0, 2, 1))/2
    skew = (jacobian-jacobian.transpose(0, 2, 1))/2
    eigen = np.linalg.eigvalsh(sym)
    difference = prediction-oracle
    return dict(field_mse=float(np.mean(difference**2)),
        velocity_jacobian_error=float(np.mean(np.sum((jacobian-oracle_jacobian)**2, axis=(1, 2))/dim)),
        image_jacobian_error=float((1-t)**2*np.mean(np.sum((jacobian-oracle_jacobian)**2, axis=(1, 2))/dim)),
        velocity_skew_energy=float(np.mean(np.sum(skew**2, axis=(1, 2))/dim)),
        image_skew_energy=float((1-t)**2*np.mean(np.sum(skew**2, axis=(1, 2))/dim)),
        image_symmetric_negative_energy=float(np.mean(np.sum(np.minimum(eigen, 0)**2, axis=1)/dim)),
        image_symmetric_negative_fraction=float(np.mean(eigen < -1e-10)),
        image_symmetric_min_eigenvalue=float(eigen.min()),
        image_symmetric_eigenvalue_quantiles=np.quantile(eigen, [.05, .5, .95]).tolist(),
        mean_field_bias_energy=float(np.mean(difference.mean(axis=0)**2)))
