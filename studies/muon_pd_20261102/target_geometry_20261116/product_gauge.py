"""Exact Product16 marginal-preserving flow with nonzero velocity error."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.linalg import expm
from scipy.special import ndtr, ndtri
from geometry import read, dump, sha
from product_quantile import marginal, inverse


def path(law, source: np.ndarray, time: float, omega: np.ndarray, coefficient: float) -> np.ndarray:
    gaussian = source@expm(coefficient*time*omega).T
    return inverse(law, ndtr(gaussian), time)


def error_field(law, point: np.ndarray, time: float, omega: np.ndarray) -> dict:
    target = marginal(law, point, time)
    gaussian = ndtri(target['cdf'])
    assert np.isfinite(gaussian).all()
    gain = np.exp(-.5*gaussian**2-.5*np.log(2*np.pi)-target['log_density'])
    speed = gaussian@omega.T
    error = gain*speed
    c = (1-time)**2+time*time*law.variances
    score = (-point+time*law.locations*np.tanh(time*law.locations*point/c))/c
    divergence = ((-gaussian-gain*score)*speed).sum(1)
    alignment = (score*error).sum(1)
    return dict(error=error, oracle=target['velocity'], divergence=divergence,
                alignment=alignment, weighted_divergence=divergence+alignment)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--quantile', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root, out = Path(a.quantile), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    complete, summary = read(root/'COMPLETE.json'), read(root/'summary.json')
    assert complete['passed'] and complete['oracle_only']
    assert sha(root/'summary.json') == complete['summary_sha256']
    manifest = Path(summary['data_manifest'])
    assert sha(manifest) == summary['data_manifest_sha256']
    data = read(manifest)['datasets']['rotated_product_bimodal_16d']['distribution']
    law = SimpleNamespace(**{k: np.asarray(v) for k, v in data.items()})
    assert sha(root/'exact-oracle.npz') == summary['analytic_arrays_sha256']
    with np.load(root/'exact-oracle.npz') as z:
        source = z['source'].copy()
        oracle_world = z['world_path'][-1].copy()
    latent_source = source@law.rotation
    dim = latent_source.shape[1]
    cyclic = np.roll(np.eye(dim), 1, axis=0)
    omega = .2*(cyclic-cyclic.T)
    assert np.array_equal(omega, -omega.T)
    assert np.linalg.norm(omega, 2) <= .4+1e-14
    oracle = inverse(law, ndtr(latent_source), 1.)
    assert np.allclose(oracle@law.rotation.T, oracle_world, atol=2e-12, rtol=2e-12)
    gain = np.exp(-.5*latent_source**2-.5*np.log(2*np.pi)-marginal(law, oracle, 1.)['log_density'])
    tangent = gain*(latent_source@omega.T)
    coefficients = [.125, .25, .5, 1.]
    endpoints, rows = [], []
    for coefficient in coefficients:
        orthogonal = expm(coefficient*omega)
        assert np.max(np.abs(orthogonal.T@orthogonal-np.eye(dim))) < 1e-14
        endpoint = path(law, latent_source, 1., omega, coefficient)
        endpoints.append(endpoint)
        displacement = endpoint-oracle
        residual = displacement-coefficient*tangent
        signal, remainder = float(np.mean(displacement**2)), float(np.mean(residual**2))
        rows.append(dict(coefficient=coefficient, displacement_mse=signal,
            remainder_mse=remainder, explained_fraction=1-remainder/signal,
            exact_population_marginals='equal to native law at every time by Gaussian rotation pushforward'))
    controls = []
    for time in [.1, .3, .5, .7, .9]:
        point = path(law, latent_source, time, omega, 1.)
        field = error_field(law, point, time, omega)
        relative_identity = np.abs(field['weighted_divergence'])/(1+np.abs(field['divergence'])+np.abs(field['alignment']))
        assert relative_identity.max() < 1e-12
        derivatives = []
        for epsilon in [1e-5, 1e-6]:
            diagonal = np.zeros(len(point))
            for axis in range(dim):
                offset = np.zeros_like(point)
                offset[:, axis] = epsilon
                diagonal += (error_field(law, point+offset, time, omega)['error'][:, axis]
                             -error_field(law, point-offset, time, omega)['error'][:, axis])/(2*epsilon)
            normalized = np.abs(diagonal-field['divergence'])/(1+np.abs(field['divergence']))
            derivatives.append(dict(epsilon=epsilon, maximum_relative_error=float(normalized.max()),
                rms_relative_error=float(np.sqrt(np.mean(normalized**2)))))
        assert min(r['maximum_relative_error'] for r in derivatives) < 1e-3
        epsilon = 1e-5
        velocity_fd = (path(law, latent_source, time+epsilon, omega, 1.)
                       -path(law, latent_source, time-epsilon, omega, 1.))/(2*epsilon)
        normalized = np.abs(velocity_fd-field['oracle']-field['error'])/(1+np.abs(velocity_fd))
        assert normalized.max() < 1e-4
        controls.append(dict(time=time, weighted_divergence_relative_max=float(relative_identity.max()),
            divergence_finite_differences=derivatives, flow_velocity_relative_max=float(normalized.max()),
            nonzero_velocity_error_mse=float(np.mean(field['error']**2))))
    # The exact endpoint derivative is checked as lambda tends to zero;
    # the four registered finite lambdas are not assumed to be in that regime.
    derivative_checks = []
    for epsilon in [1e-5, 1e-6]:
        fd = (path(law, latent_source, 1., omega, epsilon)
              -path(law, latent_source, 1., omega, -epsilon))/(2*epsilon)
        normalized = np.abs(fd-tangent)/(1e-8+np.abs(tangent))
        derivative_checks.append(dict(epsilon=epsilon, maximum_relative_error=float(normalized.max()),
            rms_relative_error=float(np.sqrt(np.mean(normalized**2)))))
    assert min(r['maximum_relative_error'] for r in derivative_checks) < 1e-3
    np.savez_compressed(out/'control.npz', source=source, omega=omega, coefficients=coefficients,
        oracle_endpoint=oracle, tangent=tangent, endpoints=np.stack(endpoints))
    result = dict(passed=True, sources=256, dimensions=dim, rows=rows, controls=controls,
        endpoint_derivative_checks=derivative_checks, matrix_norm=float(np.linalg.norm(omega, 2)),
        input_complete_sha256=sha(root/'COMPLETE.json'), raw_sha256=sha(out/'control.npz'),
        source_sha256={f.name: sha(f) for f in [Path(__file__), Path(__file__).with_name('product_quantile.py')]},
        interpretation='Exact population marginals; finite-source paired displacement and residual only, no SW2 ranking')
    dump(out/'summary.json', result)
    dump(out/'COMPLETE.json', dict(passed=True, oracle_control_only=True, sources=256,
        summary_sha256=sha(out/'summary.json'), source_sha256=sha(Path(__file__))))
    print('PASS actual-law gauge control:', rows, flush=True)


if __name__ == '__main__':
    main()
