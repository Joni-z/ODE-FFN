"""Exact latent Product16 oracle flow and its variational transition."""
import numpy as np
from scipy.special import ndtr, logsumexp


def marginal(law, position: np.ndarray, time: float) -> dict:
    if not 0 <= time <= 1:
        raise ValueError('Time must lie in [0,1]')
    c = (1-time)**2+time*time*law.variances
    width = np.sqrt(c)
    normalized = np.stack([(position-time*law.locations)/width,
                           (position+time*law.locations)/width], axis=0)
    log_component = -.5*normalized**2-.5*np.log(2*np.pi)
    logp = logsumexp(log_component, axis=0)-np.log(2*width)
    derivative_width_over_width = (time*law.variances-(1-time))/c
    component_velocity = np.stack([law.locations+derivative_width_over_width*(position-time*law.locations),
                                  -law.locations+derivative_width_over_width*(position+time*law.locations)], axis=0)
    probability = np.exp(log_component-logsumexp(log_component, axis=0))
    velocity = (probability*component_velocity).sum(0)
    return dict(cdf=ndtr(normalized).mean(0), log_density=logp, velocity=velocity)


def inverse(law, probabilities: np.ndarray, time: float) -> np.ndarray:
    """Invert the exact positive marginal; no probability clipping or inputs dropped."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.all((probabilities > 0) & (probabilities < 1)):
        raise ValueError('Finite interior Gaussian probabilities are required')
    width = np.sqrt((1-time)**2+time*time*law.variances)
    extent = time*law.locations+12*width
    low = np.broadcast_to(-extent, probabilities.shape).copy()
    high = -low.copy()
    assert np.all(marginal(law, low, time)['cdf'] < probabilities)
    assert np.all(marginal(law, high, time)['cdf'] > probabilities)
    for _ in range(64):
        middle = (low+high)/2
        cdf = marginal(law, middle, time)['cdf']
        low = np.where(cdf < probabilities, middle, low)
        high = np.where(cdf >= probabilities, middle, high)
    value = (low+high)/2
    # Symmetry fixes the median uniquely even when floating CDF evaluation
    # rounds to .5 across a numerically flat interval between separated modes.
    value = np.where(probabilities == .5, 0., value)
    assert np.max(np.abs(marginal(law, value, time)['cdf']-probabilities)) < 2e-14
    return value


def latent_path(law, source: np.ndarray, times: np.ndarray) -> np.ndarray:
    probability = ndtr(source@law.rotation)
    return np.stack([inverse(law, probability, float(t)) for t in times])


def transition(law, path: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Diagonal latent transition from each recorded time to the endpoint."""
    final = marginal(law, path[-1], float(times[-1]))['log_density']
    return np.stack([np.exp(marginal(law, point, float(t))['log_density']-final)
                     for point, t in zip(path, times)])
