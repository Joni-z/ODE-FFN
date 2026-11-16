"""Frozen Product16 marginal, component-width and dependence case study."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr, expit
from verify_density import read, sha, npz, law_for
from analyze_formal import sliced_w2


def inverse_cdf(law, u):
    n = len(u)
    sigma = np.sqrt(law.variances)
    lo = np.broadcast_to(-law.locations-12*sigma, (n, law.dim)).copy()
    hi = -lo.copy()
    for _ in range(64):
        mid = (lo+hi)/2
        cdf = .5*(ndtr((mid-law.locations)/sigma)+ndtr((mid+law.locations)/sigma))
        lo, hi = np.where(cdf < u, mid, lo), np.where(cdf >= u, mid, hi)
    q = (lo+hi)/2
    cdf = .5*(ndtr((q-law.locations)/sigma)+ndtr((q+law.locations)/sigma))
    assert np.max(np.abs(cdf-u)) < 1e-12
    return q


def quantiles(law, n):
    q = inverse_cdf(law, (np.arange(n)+.5)[:, None]/n)
    assert np.max(np.abs(q+q[::-1])) < 1e-12
    return q


def geometry(x, law, target_quantiles):
    z = x@law.rotation
    sigma = np.sqrt(law.variances)
    r = expit(2*law.locations*z/law.variances)
    weights = np.stack([1-r, r], 2)
    signs = np.array([-1., 1.])
    residual = (z[:, :, None]-law.locations[None, :, None]*signs)/sigma[None, :, None]
    mass = weights.mean(0)
    means = (weights*residual).mean(0)/mass
    second = (weights*residual**2).mean(0)/mass
    variance = second-means**2
    assert (variance >= -1e-12).all()
    mean_energy = float((mass*means**2).sum()/law.dim)
    variance_mean = float((mass*variance).sum()/law.dim)
    mass_term = float((mass*np.log(2*mass)).sum())
    centroid_term = float(.5*(mass*means**2).sum())
    width_term = float(.5*(mass*(variance-1-np.log(variance))).sum())
    energy_per_source = (weights*residual**2).sum((1, 2))/law.dim
    assert abs(mean_energy+variance_mean-float(energy_per_source.mean())) < 1e-12
    posterior_sign = 2*r-1
    moment = posterior_sign.T@posterior_sign/len(z)
    centered = posterior_sign-posterior_sign.mean(0)
    covariance = centered.T@centered/len(z)
    off = ~np.eye(law.dim, dtype=bool)
    axis_w2 = np.sqrt(((np.sort(z, axis=0)-target_quantiles)**2).mean(0))
    return dict(normalized_centroid_energy=mean_energy, normalized_conditional_variance=variance_mean,
        moment_kl_lower_bound_plugin=mass_term+centroid_term+width_term,
        moment_kl_terms=dict(mass=mass_term, centroid=centroid_term, width=width_term),
        normalized_component_second_moment=float(energy_per_source.mean()),
        component_second_moment_mc_se=float(energy_per_source.std(ddof=1)/np.sqrt(len(z))),
        posterior_sign_pair_moment_rms=float(np.sqrt(np.mean(moment[off]**2))),
        posterior_sign_pair_covariance_rms=float(np.sqrt(np.mean(covariance[off]**2))),
        plus_mass_bias_rms=float(np.sqrt(np.mean((mass[:, 1]-.5)**2))),
        latent_axis_w2_rms=float(np.sqrt(np.mean(axis_w2**2))), latent_axis_w2=axis_w2.tolist(),
        component_normalized_means=means.tolist(), component_normalized_variances=variance.tolist())


def main():
    p = argparse.ArgumentParser()
    for key in ['root', 'data', 'verification', 'output']:
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    root, data, verification, output = map(Path, [a.root, a.data, a.verification, a.output])
    assert not output.exists()
    checked = read(verification/'COMPLETE.json')
    assert checked['passed']
    cases = [c for c in read(verification/'verified-cases.json') if c['dataset'] == 'rotated_product_bimodal_16d']
    assert len(cases) == 4 and len({c['family'] for c in cases}) == 4
    manifest, meta = read(root/'portable.json'), read(data/'density-data.json')
    case = next(c for c in manifest['cases'] if c['id'] == cases[0]['id'])
    law = law_for(root, case)
    asset = meta['datasets'][case['dataset']]
    assert sha(data/asset['file']) == asset['sha256']
    arrays = npz(data/asset['file'])
    target = np.concatenate([arrays[f'target_{b}'] for b in range(4)])
    q = quantiles(law, 1024)
    calibration = geometry(target, law, q)
    source = np.concatenate([arrays[f'source_{b}'] for b in range(4)])
    # Independent Gaussian latent coordinates -> exact target mixture quantiles.
    # This is an endpoint-law control, not the native FM oracle trajectory.
    exact_control = inverse_cdf(law, ndtr(source@law.rotation))@law.rotation.T
    calibration['exact_law_endpoint_sw2_pool'] = sliced_w2(exact_control, target, arrays['directions'])
    calibration['exact_law_endpoint_sw2_banks'] = [sliced_w2(exact_control[b*256:(b+1)*256],
        arrays[f'target_{b}'], arrays['directions']) for b in range(4)]
    rows = []
    for c in cases:
        folder = Path(c['folder'])
        assert sha(folder/'COMPLETE.json') == c['completion_sha256']
        assert read(folder/'COMPLETE.json')['density_data_sha256'] == sha(data/'density-data.json')
        for cell in c['cells']:
            assert cell['qualified']
            assert sha(folder/cell['file']) == cell['sha256']
            z = npz(folder/cell['file'])
            assert z['generated'].shape == (1024, 16) and z['qualified'].all()
            value = dict(id=c['id'], family=c['family'], mode=cell['mode'],
                mean_log_ratio=float(z['log_ratio_source'].mean()), **geometry(z['generated'], law, q))
            rows.append(value)
            print(json.dumps({k: value[k] for k in ['family', 'mode', 'mean_log_ratio',
                'normalized_centroid_energy', 'normalized_conditional_variance',
                'posterior_sign_pair_covariance_rms', 'latent_axis_w2_rms']}), flush=True)
    result = dict(passed=True, exploratory_frozen_geometry=True, sources_per_model_field=1024,
        exact_population_reference=dict(normalized_centroid_energy=0., normalized_conditional_variance=1.,
            normalized_component_second_moment=1., posterior_sign_pair_covariance=0.),
        native_target_calibration=calibration, rows=rows,
        verification_sha256=sha(verification/'COMPLETE.json'), data_sha256=sha(data/'density-data.json'),
        source_sha256=sha(__file__), confirmation_time_profiles_exported=False)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
