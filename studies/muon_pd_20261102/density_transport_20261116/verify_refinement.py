"""Recompute every conditional refinement gate independently in NumPy."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from verify_density import REPO, read, sha, npz, law_for


def gates(coarse, fine, dim):
    metrics = dict(nonpositive_steps=int(fine['nonpositive_steps']),
        chain=float(np.max(np.abs(fine['chain_residual']))/dim),
        logdet=float(np.max(np.abs(fine['log_ratio_divergence']-fine['log_ratio_discrete_local']))/dim),
        ratio_refinement=max(float(np.max(np.abs(fine[k]-coarse[k]))/dim) for k in ['log_ratio_source', 'log_ratio_divergence']),
        endpoint_refinement=float(np.max(np.linalg.norm(fine['generated']-coarse['generated'], axis=1)/(1+np.linalg.norm(fine['generated'], axis=1)))))
    passed = (metrics['nonpositive_steps'] == 0 and metrics['chain'] <= .001 and metrics['logdet'] <= .001
              and metrics['ratio_refinement'] <= .002 and metrics['endpoint_refinement'] <= .001)
    return metrics, passed


def main():
    p = argparse.ArgumentParser()
    for name in ['root', 'data', 'pilot', 'verification', 'runtime', 'output']:
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    root, data_root, pilot, verified, runtime, out = map(Path, [a.root, a.data, a.pilot, a.verification, a.runtime, a.output])
    terminal = read(runtime/'queue-finished.json')
    assert terminal['passed'] and terminal['phase'] == 'refinement' and len(terminal['completed']) == 8 and not terminal['unstarted']
    meta = read(data_root/'density-data.json')
    assert terminal['density_data_sha256'] == sha(data_root/'density-data.json')
    original = read(verified/'verified-cases.json')
    maximum, new_rows, final, bindings = 0., [], [], []
    for job in terminal['completed']:
        assert job['complete'] and job['exit_code'] == 0
        case = next(c for c in read(root/'portable.json')['cases'] if c['id'] == job['id'])
        prior = next(v for v in original if v['id'] == job['id'])
        folder = runtime/'cases'/case['id']
        done = read(folder/'COMPLETE.json')
        assert done['passed'] and done['refinement'] and not done['witness'] and done['parameters_unchanged']
        assert done['density_data_sha256'] == sha(data_root/'density-data.json')
        assert done['verification_sha256'] == sha(verified/'COMPLETE.json')
        assert done['pilot_completion_sha256'] == prior['completion_sha256'] == sha(pilot/'cases'/case['id']/'COMPLETE.json')
        assert done['checkpoint_sha256'] == case['checkpoint_sha256'] == sha(root/case['checkpoint'])
        for name, value in done['source_sha256'].items():
            content = subprocess.check_output(['git', '-C', str(REPO), 'show', terminal['frozen_commit']+':studies/muon_pd_20261102/density_transport_20261116/'+name])
            assert hashlib.sha256(content).hexdigest() == value
        expected_modes = [q['mode'] for q in prior['qualifications'] if not q['qualified']]
        assert done['selected_modes'] == expected_modes
        assert len(done['decisions']) == len(expected_modes)
        data = npz(data_root/meta['datasets'][case['dataset']]['file'])
        law = law_for(root, case)
        initial = -.5*(np.sum(data['source_0'][:8]**2, axis=1)+law.dim*math.log(2*math.pi))
        old = read(pilot/'cases'/case['id']/'COMPLETE.json')
        for q in prior['qualifications']:
            if q['qualified']:
                final.append(dict(q, final_steps=512))
                continue
            rows = [r for r in done['rows'] if r['mode'] == q['mode']]
            assert [r['steps'] for r in rows] in [[1024], [1024, 2048]]
            reference = next(r for r in old['rows'] if r['mode'] == q['mode'] and r['steps'] == 512)
            file = pilot/'cases'/case['id']/reference['file']
            assert sha(file) == reference['sha256']
            coarse, previous, accepted = npz(file), 512, False
            for row in rows:
                assert not accepted and row['previous_steps'] == previous and row['native_endpoints_exact']
                assert sha(folder/row['file']) == row['sha256']
                fine = npz(folder/row['file'])
                assert fine['generated'].shape == (8, law.dim) and fine['time_parts'].shape == (8, 8, 3)
                assert all(np.isfinite(v).all() for v in fine.values())
                expected = [law.log_density(fine['generated'], 1.), -fine['time_parts'][:, :, :2].sum((1, 2)),
                    initial-fine['integral_divergence']-law.log_density(fine['generated'], 1.),
                    initial-fine['integral_local_logdet']-law.log_density(fine['generated'], 1.),
                    fine['log_ratio_source']-fine['log_ratio_divergence']]
                for key, value in zip(['endpoint_log_target', 'log_ratio_source', 'log_ratio_divergence', 'log_ratio_discrete_local', 'chain_residual'], expected):
                    error = float(np.max(np.abs(fine[key]-value)))
                    maximum = max(maximum, error)
                    assert error < 1e-9
                actual, accepted = gates(coarse, fine, law.dim)
                for key, value in actual.items():
                    error = abs(float(value)-float(row['metrics'][key]))
                    maximum = max(maximum, error)
                    assert error < 1e-12
                assert row['qualified'] == accepted
                new_rows.append(dict(id=case['id'], mode=q['mode'], steps=row['steps'], metrics=actual, qualified=accepted))
                coarse, previous = fine, row['steps']
            decision = next(v for v in done['decisions'] if v['mode'] == q['mode'])
            assert decision['qualified'] == accepted and decision['final_steps'] == previous
            final.append(dict(id=case['id'], mode=q['mode'], final_steps=previous, qualified=accepted, metrics=actual))
        assert len(done['rows']) == sum(len([r for r in done['rows'] if r['mode'] == mode]) for mode in expected_modes)
        bindings.append(dict(id=case['id'], completion_sha256=sha(folder/'COMPLETE.json')))
    assert len(final) == 16 and len({(q['id'], q['mode']) for q in final}) == 16
    out.mkdir(parents=True, exist_ok=False)
    (out/'qualified-cells.json').write_text(json.dumps(final, indent=2)+'\n')
    (out/'refined-rows.json').write_text(json.dumps(new_rows, indent=2)+'\n')
    receipt = dict(passed=True, cases=8, cells=16, qualified_cells=sum(q['qualified'] for q in final),
                   refined_rows=len(new_rows), max_independent_error=maximum, bindings=bindings,
                   source_sha256=sha(__file__), queue_receipt_sha256=sha(runtime/'queue-finished.json'))
    (out/'COMPLETE.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'bindings'}))


if __name__ == '__main__':
    main()
