"""Independent native-law/NumPy audit of every full-bank density observation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from verify_density import REPO, read, sha, npz, law_for


def verify_case(root, data_root, folder, commit, phase='formal'):
    root, data_root, folder = map(Path, [root, data_root, folder])
    manifest, meta, done = read(root/'portable.json'), read(data_root/'density-data.json'), read(folder/'COMPLETE.json')
    assert done['passed'] and done['phase'] == phase and done['parameters_unchanged'] and done['dtype'] == 'float64'
    assert done['formal'] == (phase == 'formal') and done['witness'] == (phase == 'witness')
    assert done['benchmark'] == (phase == 'benchmark')
    assert done['density_data_sha256'] == sha(data_root/'density-data.json')
    assert done['portable_manifest_sha256'] == sha(root/'portable.json') == meta['portable_manifest_sha256']
    for filename, digest in done['source_sha256'].items():
        content = subprocess.check_output(['git', '-C', str(REPO), 'show', commit+':studies/muon_pd_20261102/density_transport_20261116/'+filename])
        assert hashlib.sha256(content).hexdigest() == digest
    assert {'formal_density.py', 'density_ops.py'} <= set(done['source_sha256'])
    if done.get('derivative_backend') == 'batched':
        assert 'jacobian_backend.py' in done['source_sha256']
    case = next(c for c in manifest['cases'] if c['id'] == done['id'])
    assert done['checkpoint_sha256'] == case['checkpoint_sha256'] == sha(root/case['checkpoint'])
    assert done['dataset'] == case['dataset'] and done['family'] == case['family']
    for name in ['config', 'dataset_manifest']:
        assert sha(root/case[name]) == case[name+'_sha256']
    asset = meta['datasets'][case['dataset']]
    assert sha(data_root/asset['file']) == asset['sha256']
    data, law = npz(data_root/asset['file']), law_for(root, case)
    n = 4 if phase == 'witness' else 256
    banks = 4 if phase == 'formal' else 1
    levels = [1024, 2048] if phase == 'formal' else [16 if phase == 'witness' else 32]
    assert (done['samples_per_bank'], done['banks'], done['levels']) == (n, banks, levels)
    assert len(done['rows']) == 2*banks*len(levels)
    rows, gates, maximum = [], [], 0.
    def close(actual, expected, tolerance=1e-9):
        nonlocal maximum
        error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        maximum = max(maximum, error)
        assert error < tolerance, (case['id'], error, tolerance)
    seen = set()
    for mode in ['original', 'projected']:
        for bank in range(banks):
            source = data[f'source_{bank}'][:n]
            initial = -.5*(np.sum(source**2, axis=1)+law.dim*math.log(2*math.pi))
            values = {}
            for steps in levels:
                row = next(r for r in done['rows'] if (r['mode'], r['bank'], r['steps']) == (mode, bank, steps))
                key = mode, bank, steps
                assert key not in seen and row['native_endpoints_exact'] and row['samples'] == n
                seen.add(key)
                file = folder/row['file']
                assert sha(file) == row['sha256']
                z = npz(file)
                assert z['generated'].shape == (n, law.dim) and z['time_parts'].shape == (n, 8, 3)
                assert all(np.isfinite(v).all() for v in z.values())
                for name in ['generated', 'time_parts', 'log_ratio_source', 'log_ratio_divergence', 'log_ratio_discrete_local']:
                    assert z[name].dtype == np.float64
                logp = law.log_density(z['generated'], 1.)
                close(z['endpoint_log_target'], logp)
                close(z['log_ratio_source'], -z['time_parts'][:, :, :2].sum((1, 2)))
                close(z['log_ratio_divergence'], initial-z['integral_divergence']-logp)
                close(z['log_ratio_discrete_local'], initial-z['integral_local_logdet']-logp)
                close(z['chain_residual'], z['log_ratio_source']-z['log_ratio_divergence'])
                close(row['max_chain_residual_per_dim'], np.max(np.abs(z['chain_residual']))/law.dim)
                close(row['max_logdet_disagreement_per_dim'], np.max(np.abs(z['log_ratio_divergence']-z['log_ratio_discrete_local']))/law.dim)
                assert int(z['nonpositive_steps']) == row['nonpositive_steps']
                assert float(z['jacobian_value_error']) < 1e-10
                logratio = z['log_ratio_source']
                rows.append(dict(id=case['id'], dataset=case['dataset'], family=case['family'], mode=mode, bank=bank,
                    steps=steps, n=n, file=row['file'], sha256=row['sha256'], mean_log_ratio=float(logratio.mean()),
                    sd_log_ratio=float(logratio.std(ddof=1)), mc_se=float(logratio.std(ddof=1)/math.sqrt(n)),
                    quantiles=np.quantile(logratio, [0, .01, .5, .99, 1]).tolist(),
                    chain_max_per_dim=row['max_chain_residual_per_dim'],
                    local_logdet_max_per_dim=row['max_logdet_disagreement_per_dim'],
                    time_bin_mean=(-z['time_parts'][:, :, :2].sum(2)).mean(0).tolist(), seconds=row['seconds']))
                values[steps] = z
            if phase == 'formal':
                coarse, fine = values[1024], values[2048]
                chain = np.abs(fine['chain_residual'])/law.dim
                local = np.abs(fine['log_ratio_divergence']-fine['log_ratio_discrete_local'])/law.dim
                ratio = np.maximum(np.abs(fine['log_ratio_source']-coarse['log_ratio_source']),
                                   np.abs(fine['log_ratio_divergence']-coarse['log_ratio_divergence']))/law.dim
                endpoint = np.linalg.norm(fine['generated']-coarse['generated'], axis=1)/(1+np.linalg.norm(fine['generated'], axis=1))
                passed = (chain <= .001)&(local <= .001)&(ratio <= .002)&(endpoint <= .001)&(int(fine['nonpositive_steps']) == 0)
                gates.append(dict(id=case['id'], mode=mode, bank=bank, qualified=bool(passed.all()),
                    qualified_points=int(passed.sum()), failed_source_indices=np.flatnonzero(~passed).tolist(),
                    maximums=dict(chain=float(chain.max()), local_logdet=float(local.max()), ratio_refinement=float(ratio.max()), endpoint_refinement=float(endpoint.max())),
                    nonpositive_steps=int(fine['nonpositive_steps']),
                    mean_log_ratio_change=float((fine['log_ratio_source']-coarse['log_ratio_source']).mean())))
    assert len(seen) == len(done['rows'])
    return dict(passed=True, phase=phase, id=case['id'], device=done['device'],
        derivative_backend=done.get('derivative_backend', 'coordinate'), rows=rows, numerical_gates=gates,
        source_sha256=done['source_sha256'], completion_sha256=sha(folder/'COMPLETE.json'), max_independent_error=maximum)


def main():
    p = argparse.ArgumentParser()
    for name in ['root', 'data', 'output']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--runtimes', nargs='+')
    p.add_argument('--probe-case')
    p.add_argument('--commit')
    a = p.parse_args()
    out = Path(a.output)
    assert not out.exists()
    if a.probe_case:
        assert a.commit
        phase = read(Path(a.probe_case)/'COMPLETE.json')['phase']
        assert phase in ['witness', 'benchmark']
        result = verify_case(a.root, a.data, a.probe_case, a.commit, phase)
        out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(passed=True, phase=phase, id=result['id'], max_independent_error=result['max_independent_error'])))
        return
    assert a.runtimes
    results = []
    for runtime in map(Path, a.runtimes):
        terminal = read(runtime/'queue-finished.json')
        assert terminal['passed'] and terminal['phase'] == 'formal' and not terminal['unstarted']
        assert terminal['density_data_sha256'] == sha(Path(a.data)/'density-data.json')
        assert len(terminal['completed']) == len(terminal['selected_indices'])
        for row in terminal['completed']:
            assert row['exit_code'] == 0 and row['complete']
            if terminal['device'] == 'cuda':
                assert row['physical_gpu'] in [4, 5, 6, 7]
            folder = runtime/'cases'/row['id']
            assert read(folder/'COMPLETE.json')['gpu_uuid'] == row['gpu_uuid']
            result = verify_case(a.root, a.data, folder, terminal['frozen_commit'])
            result['raw_folder'] = str(folder)
            results.append(result)
    expected = {r['id'] for r in read(Path(a.data)/'density-data.json')['references']}
    assert len(results) == 8 and {r['id'] for r in results} == expected
    assert len({tuple(sorted(r['source_sha256'].items())) for r in results}) == 1
    rows = [v for r in results for v in r['rows']]
    gates = [v for r in results for v in r['numerical_gates']]
    assert len(rows) == 128 and len(gates) == 64
    cells = [dict(id=r['id'], mode=mode, qualified=all(g['qualified'] for g in r['numerical_gates'] if g['mode'] == mode))
             for r in results for mode in ['original', 'projected']]
    out.mkdir(parents=True)
    (out/'verified-cases.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    receipt = dict(passed=True, cases=8, rows=128, bank_gate_checks=64, resolved_cells=sum(c['qualified'] for c in cells), cells=cells,
                   max_independent_error=max(r['max_independent_error'] for r in results), source_sha256=sha(__file__))
    (out/'COMPLETE.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'cells'}))


if __name__ == '__main__':
    main()
