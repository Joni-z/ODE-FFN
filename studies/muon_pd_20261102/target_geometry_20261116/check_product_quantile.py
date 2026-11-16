"""Independent oracle-only check; does not rank incomplete learned-model results."""
import argparse
import importlib.util
from pathlib import Path
import sys
import numpy as np
from scipy.special import ndtr
from geometry import read, dump, sha
from product_quantile import marginal, inverse, latent_path, transition


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ['study', 'pilot', 'data-manifest', 'completed-case', 'output']:
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    study, pilot, manifest, case, out = map(Path, [a.study, a.pilot, a.data_manifest, a.completed_case, a.output])
    out.mkdir(parents=True, exist_ok=False)
    native_file = Path(__file__).resolve().parent.parent/'reference/synthetic/data.py'
    spec = importlib.util.spec_from_file_location('quantile_native_law', native_file)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    data = read(manifest)['datasets']['rotated_product_bimodal_16d']
    assert data['family'] == 'rotated_product_mixture'
    law = native.RotatedProductMixture(**data['distribution'])
    anchors = read(study/'anchors.json')
    asset_manifest = Path(anchors['assets'])/'manifest.json'
    asset = next(r for r in read(asset_manifest)['rows']
                 if r['dataset'] == 'rotated_product_bimodal_16d' and r['split'] == 'selection')
    assert sha(manifest) == asset['data_manifest_sha256']
    assert sha(Path(asset['file'])) == asset['sha256']
    with np.load(asset['file']) as z:
        source = np.concatenate([z[f'source_{b}'][:32] for b in range(8)]).astype(float)
    times = np.linspace(0, 1, 9)
    path = latent_path(law, source, times)
    world = path@law.rotation.T
    initial = source@law.rotation
    probabilities = ndtr(initial)
    ratios = transition(law, path, times)
    analytic_rows = []
    for time, point in zip(times, path):
        result = marginal(law, point, float(time))
        oracle = law.velocity(point@law.rotation.T, float(time))@law.rotation
        verror = float(np.max(np.abs(oracle-result['velocity'])))
        cdf_error = float(np.max(np.abs(result['cdf']-probabilities)))
        log_error = float(np.max(np.abs(result['log_density'].sum(1)-law.log_density(point@law.rotation.T, float(time)))))
        assert verror < 1e-10 and cdf_error < 2e-14 and log_error < 1e-9
        analytic_rows.append(dict(time=float(time), velocity_identity_max=verror,
            cdf_conservation_max=cdf_error, log_density_max=log_error))
    assert np.allclose(world[0], source, atol=2e-12, rtol=2e-12)
    finite = []
    for time in [.25, .5, .75, 1.]:
        exact_point = inverse(law, probabilities, time)
        derivative = np.exp(marginal(law, initial, 0.)['log_density']
                            -marginal(law, exact_point, time)['log_density'])
        errors = []
        for epsilon in [1e-4, 1e-5, 1e-6]:
            plus = inverse(law, ndtr(initial+epsilon), time)
            minus = inverse(law, ndtr(initial-epsilon), time)
            fd = (plus-minus)/(2*epsilon)
            normalized = np.abs(fd-derivative)/(1e-8+np.abs(derivative))
            errors.append(dict(epsilon=epsilon, maximum_relative_error=float(normalized.max()),
                rms_relative_error=float(np.sqrt(np.mean(normalized**2)))))
        assert min(r['maximum_relative_error'] for r in errors) < 1e-3
        finite.append(dict(time=time, errors=errors))
    time_derivatives = []
    for time in [.125, .25, .5, .75, .875]:
        point = inverse(law, probabilities, time)
        velocity = marginal(law, point, time)['velocity']
        errors = []
        for epsilon in [1e-4, 1e-5]:
            derivative = (inverse(law, probabilities, time+epsilon)
                          -inverse(law, probabilities, time-epsilon))/(2*epsilon)
            error = np.abs(derivative-velocity)
            errors.append(dict(epsilon=epsilon, maximum=float(error.max()),
                               rms=float(np.sqrt(np.mean(error**2)))))
        assert errors[-1]['maximum'] < 1e-5
        time_derivatives.append(dict(time=time, errors=errors))
    pilot_complete, pilot_rows = read(pilot/'product/COMPLETE.json'), read(pilot/'product/rows.json')
    assert pilot_complete['passed'] and not pilot_complete['witness']
    assert sha(pilot/'product/rows.json') == pilot_complete['rows_sha256']
    archived, common = [], {}
    for row in pilot_rows['rows']:
        for level in row['levels']:
            file = pilot/'product'/level['file']
            assert sha(file) == level['sha256']
            with np.load(file) as z:
                assert np.array_equal(source, z['source'])
                if level['steps'] in common:
                    assert np.array_equal(common[level['steps']], z['oracle_path'])
                else:
                    common[level['steps']] = z['oracle_path'].copy()
                error = z['oracle_path']-world
                archived.append(dict(id=row['id'], steps=level['steps'],
                    endpoint_max=float(np.max(np.abs(error[-1]))), path_max=float(np.max(np.abs(error))),
                    file=str(file), sha256=sha(file)))
    complete, rows = read(case/'COMPLETE.json'), read(case/'rows.json')
    assert complete['passed'] and not complete['witness'] and complete['sources'] == 256
    assert sha(case/'rows.json') == complete['rows_sha256']
    queue = read(case.parent.parent/'status.json')
    terminal = next(r for r in queue['completed'] if r['id'] == complete['id'])
    assert terminal['exit_code'] == 0 and terminal['complete']
    fine = []
    for level in rows['levels']:
        file = case/level['file']
        assert sha(file) == level['sha256']
        with np.load(file) as z:
            assert np.array_equal(source, z['source'])
            error = z['oracle_path']-world
            fine.append(dict(steps=level['steps'], endpoint_rms=float(np.sqrt(np.mean(error[-1]**2))),
                endpoint_max=float(np.max(np.abs(error[-1]))), path_max=float(np.max(np.abs(error))),
                every_path_component_pass=bool(np.all(np.abs(error) <= 1e-4+1e-3*np.abs(world))),
                file=str(file), sha256=sha(file)))
    np.savez_compressed(out/'exact-oracle.npz', source=source, times=times, latent_path=path,
        world_path=world, transition_to_endpoint=ratios)
    summary = dict(passed=True, sources=256, coordinates=law.dim, times=9,
        analytic_identity_checks=analytic_rows, source_derivative_checks=finite,
        inverse_cdf_time_derivative_checks=time_derivatives,
        all_six_archived_oracle_paths_identical_at_same_resolution=True,
        archived_native_checks=archived, refined_native_checks=fine,
        transition_to_endpoint_quantiles=np.quantile(ratios, [.0, .5, .9, .99, 1.]).tolist(),
        maximum_source_to_endpoint_derivative=float(ratios[0].max()),
        maximum_time_to_endpoint_derivative=float(ratios.max()),
        data_manifest=str(manifest), data_manifest_sha256=sha(manifest),
        assets_file=asset['file'], assets_sha256=asset['sha256'],
        analytic_arrays_sha256=sha(out/'exact-oracle.npz'),
        completed_reference_case=str(case), completed_reference_case_sha256=sha(case/'COMPLETE.json'),
        source_sha256={f.name: sha(f) for f in [Path(__file__), Path(__file__).with_name('product_quantile.py'), native_file]})
    dump(out/'summary.json', summary)
    dump(out/'COMPLETE.json', dict(passed=True, sources=256, oracle_only=True,
        summary_sha256=sha(out/'summary.json'), source_sha256=sha(Path(__file__))))
    print('PASS exact Product16 oracle quantile flow:', fine, flush=True)


if __name__ == '__main__':
    main()
