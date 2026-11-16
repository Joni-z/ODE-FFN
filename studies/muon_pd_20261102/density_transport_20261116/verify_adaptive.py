"""Independent NumPy reconstruction of source selection and adaptive merges."""
import argparse
import hashlib
import math
from pathlib import Path
import subprocess
import numpy as np
from verify_formal import verify_case as verify_parent
from verify_density import REPO, read, sha, npz, law_for


POINT_KEYS = ('generated', 'log_ratio_source', 'log_ratio_divergence',
              'log_ratio_discrete_local', 'chain_residual', 'time_parts',
              'integral_divergence', 'integral_local_logdet', 'endpoint_log_target')


def verify_case(root, data_root, parent, folder, source_commit):
    root, data_root, parent, folder = map(Path, [root, data_root, parent, folder])
    done, old = read(folder/'COMPLETE.json'), read(parent/'COMPLETE.json')
    assert done['passed'] and done['adaptive'] and done['parameters_unchanged']
    assert done['id'] == old['id']
    assert old['phase'] == 'formal' or done['witness']
    prior = verify_parent(root, data_root, parent, done['parent_commit'], old['phase'])
    assert prior['completion_sha256'] == done['parent_completion_sha256']
    assert done['density_data_sha256'] == sha(data_root/'density-data.json')
    for name, digest in done['source_sha256'].items():
        blob = subprocess.check_output(['git', '-C', str(REPO), 'show', source_commit+
            ':studies/muon_pd_20261102/density_transport_20261116/'+name])
        assert hashlib.sha256(blob).hexdigest() == digest
    case = next(c for c in read(root/'portable.json')['cases'] if c['id'] == done['id'])
    assert done['checkpoint_sha256'] == case['checkpoint_sha256']
    law = law_for(root, case)
    asset = read(data_root/'density-data.json')['datasets'][case['dataset']]
    data = npz(data_root/asset['file'])
    sources = np.concatenate([data[f'source_{b}'] for b in range(4)])
    if done['witness']:
        sources = sources[:4]
    n, dim = sources.shape
    assert n == (4 if done['witness'] else 1024)
    maximum, used = 0., []
    def close(actual, expected):
        nonlocal maximum
        error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        maximum = max(maximum, error)
        assert error < 1e-9, (done['id'], error)
    def errors_and_accept(coarse, fine, nonpositive):
        e = np.empty((n, 4))
        e[:, 0] = np.abs(fine['log_ratio_source']-fine['log_ratio_divergence'])/dim
        e[:, 1] = np.abs(fine['log_ratio_divergence']-fine['log_ratio_discrete_local'])/dim
        e[:, 2] = np.maximum(np.abs(fine['log_ratio_source']-coarse['log_ratio_source']),
                             np.abs(fine['log_ratio_divergence']-coarse['log_ratio_divergence']))/dim
        e[:, 3] = np.sqrt(((fine['generated']-coarse['generated'])**2).sum(1))/(
            1+np.sqrt((fine['generated']**2).sum(1)))
        ok = (e[:, 0] <= .001)&(e[:, 1] <= .001)&(e[:, 2] <= .002)&(e[:, 3] <= .001)&(nonpositive == 0)
        return e, ok
    def subset(mode, steps, expected_indices):
        matching = [r for r in done['rows'] if (r['mode'], r['steps']) == (mode, steps)]
        assert len(matching) == 1
        row = matching[0]
        assert row['native_endpoints_exact'] and row['samples'] == len(expected_indices)
        assert sha(folder/row['file']) == row['sha256']
        z = npz(folder/row['file'])
        assert np.array_equal(z['flat_indices'], expected_indices)
        assert z['generated'].shape == (len(expected_indices), dim)
        assert z['time_parts'].shape == (len(expected_indices), 8, 3)
        assert all(np.isfinite(v).all() for v in z.values())
        for k in POINT_KEYS:
            assert z[k].dtype == np.float64
        logp = law.log_density(z['generated'], 1.)
        logsource = -.5*((sources[expected_indices]**2).sum(1)+dim*math.log(2*math.pi))
        close(z['endpoint_log_target'], logp)
        close(z['log_ratio_source'], -z['time_parts'][:, :, :2].sum((1, 2)))
        close(z['log_ratio_divergence'], logsource-z['integral_divergence']-logp)
        close(z['log_ratio_discrete_local'], logsource-z['integral_local_logdet']-logp)
        close(z['chain_residual'], z['log_ratio_source']-z['log_ratio_divergence'])
        assert int(z['nonpositive_steps']) == row['nonpositive_steps']
        assert float(z['jacobian_value_error']) < 1e-10
        used.append((mode, steps))
        return z
    cells = []
    for mode in ['original', 'projected']:
        if done['witness']:
            coarse = subset(mode, 16, np.arange(n))
            fine = subset(mode, 32, np.arange(n))
            positive_bound = np.full(n, int(fine['nonpositive_steps']))
            coarse, fine = [{k: z[k].copy() for k in POINT_KEYS} for z in [coarse, fine]]
            steps, previous_steps = np.full(n, 32), np.full(n, 16)
        else:
            levels = {}
            for level in [1024, 2048]:
                chunks = [npz(parent/f'{mode}-bank{b}-heun{level}.npz') for b in range(4)]
                levels[level] = {k: np.concatenate([z[k] for z in chunks]) for k in POINT_KEYS}
                if level == 2048:
                    positive_bound = np.concatenate([np.full(256, int(z['nonpositive_steps'])) for z in chunks])
            coarse, fine = levels[1024], levels[2048]
            steps, previous_steps = np.full(n, 2048), np.full(n, 1024)
        errors, accepted = errors_and_accept(coarse, fine, positive_bound)
        cell = next(c for c in done['cells'] if c['mode'] == mode)
        assert cell['initial_unresolved_indices'] == np.flatnonzero(~accepted).tolist()
        if not done['witness']:
            for level in [4096, 8192, 16384]:
                indices = np.flatnonzero(~accepted)
                if not len(indices):
                    break
                refined = subset(mode, level, indices)
                for k in POINT_KEYS:
                    coarse[k][indices] = fine[k][indices]
                    fine[k][indices] = refined[k]
                previous_steps[indices] = steps[indices]
                steps[indices] = level
                positive_bound[indices] = int(refined['nonpositive_steps'])
                errors, accepted = errors_and_accept(coarse, fine, positive_bound)
        assert cell['samples'] == n and sha(folder/cell['file']) == cell['sha256']
        merged = npz(folder/cell['file'])
        for k in POINT_KEYS:
            assert np.array_equal(merged[k], fine[k])
            assert np.array_equal(merged['previous_'+k], coarse[k])
        for k, expected in [('bank', np.arange(n)//256), ('source_index', np.arange(n)%256),
                            ('selected_steps', steps), ('previous_steps', previous_steps),
                            ('nonpositive_steps_bound', positive_bound), ('qualified', accepted)]:
            assert np.array_equal(merged[k], expected), k
        close(merged['errors'], errors)
        assert cell['unresolved_indices'] == np.flatnonzero(~accepted).tolist()
        assert cell['numerical_qualified'] == (bool(accepted.all()) and not done['witness'])
        assert cell['resolution_counts'] == {str(k): int((steps == k).sum()) for k in np.unique(steps)}
        close(cell['max_errors'], errors.max(0))
        cells.append(dict(mode=mode, qualified=cell['numerical_qualified'], samples=n,
            unresolved_indices=cell['unresolved_indices'], resolution_counts=cell['resolution_counts'],
            file=cell['file'], sha256=cell['sha256'], max_errors=cell['max_errors']))
    assert len(set(used)) == len(used) == len(done['rows'])
    return dict(passed=True, witness=done['witness'], id=case['id'], dataset=case['dataset'], family=case['family'],
        completion_sha256=sha(folder/'COMPLETE.json'), folder=str(folder), parent=str(parent),
        cells=cells, max_independent_error=maximum)


def main():
    p = argparse.ArgumentParser()
    for key in ['root', 'data', 'source-commit', 'output']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--parents', nargs='+', required=True)
    a = p.parse_args()
    assert len(a.cases) == len(a.parents)
    results = [verify_case(a.root, a.data, parent, case, a.source_commit)
               for case, parent in zip(a.cases, a.parents)]
    assert len({r['id'] for r in results}) == len(results)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    import json
    (out/'verified-cases.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    receipt = dict(passed=True, cases=len(results), witness=any(r['witness'] for r in results),
        qualified_cells=sum(c['qualified'] for r in results for c in r['cells']),
        max_independent_error=max(r['max_independent_error'] for r in results),
        density_data_sha256=sha(Path(a.data)/'density-data.json'), source_commit=a.source_commit,
        verification_source_sha256=sha(__file__))
    (out/'COMPLETE.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
