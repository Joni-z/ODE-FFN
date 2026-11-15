"""Independent NumPy/native-law checks; pilot qualification is not model ranking."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location('density_native_laws', HERE.parent/'reference/synthetic/data.py')
native = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = native
spec.loader.exec_module(native)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def law_for(root, case):
    value = read(root/case['dataset_manifest'])['datasets'][case['dataset']]
    types = {'gaussian_mixture': native.GaussianMixture, 'rotated_product_mixture': native.RotatedProductMixture}
    return types[value['family']](**value['distribution'])


def native_score(law, x, t):
    if hasattr(law, 'locations'):
        z = x@law.rotation
        c = (1-t)**2+t**2*law.variances
        return ((-z+t*law.locations*np.tanh(t*law.locations*z/c))/c)@law.rotation.T
    _, variance, residual, probability, _ = law._terms(x, t)
    return -np.sum(probability[:, :, None]*residual/variance, axis=1)


def field_stats(values):
    d, a, source, mse = [values[k] for k in ['divergence_error', 'alignment', 'density_source', 'field_mse']]
    return dict(field_mse=float(mse.mean()), source_rms=float(np.sqrt(np.mean(source**2))),
                divergence_error_rms=float(np.sqrt(np.mean(d**2))), alignment_rms=float(np.sqrt(np.mean(a**2))),
                source_mean=float(source.mean()), divergence_alignment_cross=float(np.mean(d*a)))


def verify_case(root, data_root, folder, expected_commit, witness=False):
    root, data_root, folder = map(Path, [root, data_root, folder])
    manifest, meta, done = read(root/'portable.json'), read(data_root/'density-data.json'), read(folder/'COMPLETE.json')
    assert done['passed'] and done['parameters_unchanged'] and done['witness'] == witness and done['pilot'] != witness
    assert done['portable_manifest_sha256'] == sha(root/'portable.json') == meta['portable_manifest_sha256']
    assert done['density_data_sha256'] == sha(data_root/'density-data.json')
    for name, digest in done['source_sha256'].items():
        content = subprocess.check_output(['git', '-C', str(REPO), 'show', expected_commit+':studies/muon_pd_20261102/density_transport_20261116/'+name])
        assert hashlib.sha256(content).hexdigest() == digest
    case = next(c for c in manifest['cases'] if c['id'] == done['id'])
    assert sha(root/case['checkpoint']) == case['checkpoint_sha256'] == done['checkpoint_sha256']
    assert sha(root/case['dataset_manifest']) == case['dataset_manifest_sha256']
    asset = meta['datasets'][case['dataset']]
    assert sha(data_root/asset['file']) == asset['sha256']
    data = npz(data_root/asset['file'])
    law = law_for(root, case)
    dim = law.dim
    points, n, levels = (16, 4, [16, 32]) if witness else (1280, 8, [128, 256, 512])
    maximum = 0.
    def close(actual, expected, tolerance=1e-9):
        nonlocal maximum
        error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        maximum = max(maximum, error)
        assert error < tolerance, (case['id'], error, tolerance)
    fields, observations, qualifications = [], [], []
    assert len(done['fields']) == 2 and len(done['rows']) == 2*len(levels)
    for mode in ['original', 'projected']:
        row = next(v for v in done['fields'] if v['mode'] == mode)
        assert sha(folder/row['file']) == row['sha256'] and row['points'] == points
        arrays = npz(folder/row['file'])
        x, t = data['field_x'][:points], data['field_t'][:points]
        assert arrays['velocity'].shape == (points, dim)
        for k in ['divergence_error', 'alignment', 'density_source', 'field_mse']:
            assert arrays[k].shape == (points,) and np.isfinite(arrays[k]).all()
        error = arrays['velocity']-law.velocity(x, t)
        close(arrays['alignment'], np.sum(error*native_score(law, x, t), axis=1))
        close(arrays['field_mse'], np.mean(error**2, axis=1))
        close(arrays['density_source'], arrays['divergence_error']+arrays['alignment'])
        profiles = []
        for clock in sorted(set(data['field_clocks'][:points])):
            mask = data['field_clocks'][:points] == clock
            profiles.append(dict(clock=int(clock), time=float(t[mask][0, 0]), points=int(mask.sum()),
                                 **field_stats({k: v[mask] for k, v in arrays.items()})))
        fields.append(dict(id=case['id'], mode=mode, profiles=profiles, **field_stats(arrays)))
        values = {}
        source = data['source_0'][:n]
        initial = -.5*(np.sum(source**2, axis=1)+dim*math.log(2*math.pi))
        for steps in levels:
            row = next(v for v in done['rows'] if v['mode'] == mode and v['steps'] == steps)
            assert row['base_points'] == n and row['native_endpoints_exact'] and sha(folder/row['file']) == row['sha256']
            z = npz(folder/row['file'])
            assert z['generated'].shape == (n, dim) and z['time_parts'].shape == (n, 8, 3)
            assert all(np.isfinite(v).all() for v in z.values())
            logp = law.log_density(z['generated'], 1.)
            close(z['endpoint_log_target'], logp)
            close(z['log_ratio_source'], -z['time_parts'][:, :, :2].sum((1, 2)))
            close(z['log_ratio_divergence'], initial-z['integral_divergence']-logp)
            close(z['log_ratio_discrete_local'], initial-z['integral_local_logdet']-logp)
            close(z['chain_residual'], z['log_ratio_source']-z['log_ratio_divergence'])
            close(row['max_chain_residual_per_dim'], np.max(np.abs(z['chain_residual']))/dim)
            close(row['max_logdet_disagreement_per_dim'], np.max(np.abs(z['log_ratio_divergence']-z['log_ratio_discrete_local']))/dim)
            assert int(z['nonpositive_steps']) == row['nonpositive_steps']
            assert float(z['jacobian_value_error']) < 1e-10
            values[steps] = z
            observations.append(dict(id=case['id'], mode=mode, steps=steps,
                max_chain_residual_per_dim=row['max_chain_residual_per_dim'],
                max_logdet_disagreement_per_dim=row['max_logdet_disagreement_per_dim'],
                nonpositive_steps=row['nonpositive_steps'], seconds=row['seconds']))
        if not witness:
            coarse, fine = values[256], values[512]
            metrics = dict(nonpositive_steps=int(fine['nonpositive_steps']),
                chain=float(np.max(np.abs(fine['chain_residual']))/dim),
                logdet=float(np.max(np.abs(fine['log_ratio_divergence']-fine['log_ratio_discrete_local']))/dim),
                ratio_refinement=max(float(np.max(np.abs(fine[k]-coarse[k]))/dim)
                                     for k in ['log_ratio_source', 'log_ratio_divergence']),
                endpoint_refinement=float(np.max(np.linalg.norm(fine['generated']-coarse['generated'], axis=1)/(1+np.linalg.norm(fine['generated'], axis=1)))))
            gates = dict(positive_steps=metrics['nonpositive_steps'] == 0, chain=metrics['chain'] <= .001,
                         local_logdet=metrics['logdet'] <= .001, ratio_refinement=metrics['ratio_refinement'] <= .002,
                         endpoint_refinement=metrics['endpoint_refinement'] <= .001)
            qualifications.append(dict(id=case['id'], mode=mode, metrics=metrics, gates=gates, qualified=all(gates.values())))
    return dict(passed=True, witness=witness, id=case['id'], field_summaries=fields, observations=observations,
                qualifications=qualifications, max_independent_error=maximum, completion_sha256=sha(folder/'COMPLETE.json'))


def main():
    p = argparse.ArgumentParser()
    for name in ['root', 'data', 'runtime', 'output']:
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    runtime, out = Path(a.runtime), Path(a.output)
    terminal = read(runtime/'queue-finished.json')
    assert terminal['passed'] and len(terminal['completed']) == 8 and not terminal['unstarted']
    assert terminal['density_data_sha256'] == sha(Path(a.data)/'density-data.json')
    results = []
    for row in terminal['completed']:
        assert row['exit_code'] == 0 and row['complete']
        results.append(verify_case(a.root, a.data, runtime/'cases'/row['id'], terminal['frozen_commit']))
    assert len({r['id'] for r in results}) == 8
    out.mkdir(parents=True, exist_ok=False)
    (out/'verified-cases.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    qualifications = [q for r in results for q in r['qualifications']]
    receipt = dict(passed=True, cases=8, measured_cells=16, resolution_rows=48,
                   qualified_cells=sum(q['qualified'] for q in qualifications),
                   max_independent_error=max(r['max_independent_error'] for r in results),
                   source_sha256=sha(__file__), queue_receipt_sha256=sha(runtime/'queue-finished.json'))
    (out/'COMPLETE.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
