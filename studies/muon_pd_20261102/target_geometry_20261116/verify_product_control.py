"""Reconstruct actual-law control from raw arrays without its implementation."""
import argparse
import math
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.linalg import expm
from geometry import read, dump, sha


def density_cdf(position: np.ndarray, locations: np.ndarray, variances: np.ndarray, time: float) -> tuple:
    width = np.sqrt((1-time)**2+time*time*variances)
    positive = (position-time*locations)/width
    negative = (position+time*locations)/width
    density = (np.exp(-.5*positive**2)+np.exp(-.5*negative**2))/(2*np.sqrt(2*np.pi)*width)
    return density, (ndtr(positive)+ndtr(negative))/2


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ['quantile', 'gauge', 'output']:
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    q, g, out = map(Path, [a.quantile, a.gauge, a.output])
    out.mkdir(parents=True, exist_ok=False)
    qc, gc = read(q/'COMPLETE.json'), read(g/'COMPLETE.json')
    assert qc['passed'] and gc['passed'] and qc['sources'] == gc['sources'] == 256
    assert sha(q/'summary.json') == qc['summary_sha256'] and sha(g/'summary.json') == gc['summary_sha256']
    qs, gs = read(q/'summary.json'), read(g/'summary.json')
    assert gs['input_complete_sha256'] == sha(q/'COMPLETE.json')
    manifest = Path(qs['data_manifest'])
    assert sha(manifest) == qs['data_manifest_sha256']
    law = read(manifest)['datasets']['rotated_product_bimodal_16d']['distribution']
    locations, variances, rotation = [np.asarray(law[k]) for k in ['locations', 'variances', 'rotation']]
    assert sha(q/'exact-oracle.npz') == qs['analytic_arrays_sha256']
    assert sha(g/'control.npz') == gs['raw_sha256']
    with np.load(q/'exact-oracle.npz') as z:
        source, latent, world, ratios, times = [z[k].copy() for k in ['source', 'latent_path', 'world_path', 'transition_to_endpoint', 'times']]
    original = source@rotation
    assert source.shape == (256, 16) and latent.shape == (9, 256, 16)
    assert np.array_equal(times, np.linspace(0, 1, 9))
    assert np.max(np.abs(world-latent@rotation.T)) == 0
    final_density = density_cdf(latent[-1], locations, variances, 1.)[0]
    cdf_errors, ratio_errors = [], []
    for i, t in enumerate(times):
        density, cdf = density_cdf(latent[i], locations, variances, float(t))
        cdf_errors.append(float(np.max(np.abs(cdf-ndtr(original)))))
        ratio_errors.append(float(np.max(np.abs(density/final_density-ratios[i]))))
    assert max(cdf_errors) < 2e-14 and max(ratio_errors) < 1e-9
    with np.load(g/'control.npz') as z:
        assert np.array_equal(z['source'], source)
        omega, oracle, tangent, endpoints, coefficients = [z[k].copy() for k in ['omega', 'oracle_endpoint', 'tangent', 'endpoints', 'coefficients']]
    cyclic = np.roll(np.eye(16), 1, axis=0)
    assert np.array_equal(omega, .2*(cyclic-cyclic.T)) and np.array_equal(omega, -omega.T)
    assert np.array_equal(coefficients, [.125, .25, .5, 1.])
    assert endpoints.shape == (4, 256, 16) and np.array_equal(oracle, latent[-1])
    gaussian_density = np.exp(-.5*original**2)/np.sqrt(2*np.pi)
    derivative = gaussian_density/final_density*(original@omega.T)
    derivative_error = float(np.max(np.abs(derivative-tangent)))
    assert derivative_error < 1e-9
    rows, reconstruction_error = [], 0.
    for i, coefficient in enumerate(coefficients):
        orthogonal = expm(float(coefficient)*omega)
        assert np.max(np.abs(orthogonal.T@orthogonal-np.eye(16))) < 1e-14
        expected_cdf = ndtr(original@orthogonal.T)
        endpoint_cdf = density_cdf(endpoints[i], locations, variances, 1.)[1]
        assert np.max(np.abs(endpoint_cdf-expected_cdf)) < 2e-14
        displacement = endpoints[i]-oracle
        residual = displacement-coefficient*derivative
        signal = math.fsum(float(x*x) for x in displacement.ravel())/displacement.size
        remainder = math.fsum(float(x*x) for x in residual.ravel())/residual.size
        fraction = 1-remainder/signal
        row = dict(coefficient=float(coefficient), displacement_mse=signal,
                   remainder_mse=remainder, explained_fraction=fraction)
        for key in row:
            reconstruction_error = max(reconstruction_error, abs(row[key]-gs['rows'][i][key]))
            assert abs(row[key]-gs['rows'][i][key]) < 1e-9
        rows.append(row)
    result = dict(passed=True, sources=256, coordinates=16, rows=rows,
        quantile_cdf_conservation_max=max(cdf_errors), transition_reconstruction_max=max(ratio_errors),
        tangent_reconstruction_max=derivative_error, metric_reconstruction_max=reconstruction_error,
        quantile_input_complete_sha256=sha(q/'COMPLETE.json'), gauge_input_complete_sha256=sha(g/'COMPLETE.json'),
        source_sha256=sha(Path(__file__)))
    dump(out/'summary.json', result)
    dump(out/'COMPLETE.json', dict(passed=True, sources=256, summary_sha256=sha(out/'summary.json'),
        source_sha256=sha(Path(__file__))))
    print('PASS independent actual-law control', result, flush=True)


if __name__ == '__main__':
    main()
