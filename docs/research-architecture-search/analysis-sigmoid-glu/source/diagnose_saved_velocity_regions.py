"""Read-only Stage17 Product16 velocity attribution using preregistered entropy bins."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs/research-architecture-search'
STUDY = Path('/Users/mr.z/ode-workspace-prep/architecture-search/shared_features_20261024_m1/output/shared_features_product16_20261024_m1')
DATA = Path('/Users/mr.z/ode-workspace-prep/data/synthetic/product16_v1')
DATASET = 'rotated_product_bimodal_16d'
ARMS = ('vector_parent', 'swiglu_budget', 'swiglu')
TIMES = (.1, .3, .5, .7, .9)
EDGES = np.array([0., .1, .5, .9, 1.])
REGIONS = ('[0,.1)', '[.1,.5)', '[.5,.9)', '[.9,1]')
PROTOCOL_SHA = '399c89307b462ad91698c0b3621826295af66b4ac827e207025171e01d450251'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = DOCS / 'protocol-posterior-regions.md'
    assert sha(protocol) == PROTOCOL_SHA
    manifest = json.loads((DATA / 'manifest.json').read_text())
    spec = manifest['datasets'][DATASET]['distribution']
    Q, mu, var = (np.asarray(spec[k], dtype=np.float64)
                  for k in ('rotation', 'locations', 'variances'))
    np.testing.assert_allclose(Q.T @ Q, np.eye(16), atol=1e-12, rtol=0)
    files = {str(DATA / 'manifest.json'): sha(DATA / 'manifest.json')}
    files[str(DATA / DATASET / 'validation.npz')] = sha(DATA / DATASET / 'validation.npz')
    common, errors, summaries, bins = {}, {}, {}, {}
    max_oracle_error, max_rotation_mse_error = 0., 0.
    for arm in ARMS:
        directory = STUDY / 'diagnostics' / DATASET / arm
        summary_path = directory / 'summary.json'
        summary = json.loads(summary_path.read_text())
        summaries[arm] = summary
        files[str(summary_path)] = sha(summary_path)
        assert summary['manifest_sha256'] == files[str(DATA / 'manifest.json')]
        assert summary['targets_sha256'] == files[str(DATA / DATASET / 'validation.npz')]
        assert summary['step'] == 5000 and summary['seed'] == 0
        for t in TIMES:
            path = directory / f'common-{t}.npz'
            files[str(path)] = sha(path)
            assert files[str(path)] == summary['arrays_sha256'][path.name]
            with np.load(path, allow_pickle=False) as bank:
                x, pred, oracle = (bank[k] for k in ('x', 'prediction', 'oracle'))
                assert x.shape == pred.shape == oracle.shape == (512, 16)
                assert all(z.dtype == np.float64 and np.isfinite(z).all() for z in (x, pred, oracle))
                if t in common:
                    np.testing.assert_array_equal(x, common[t][0])
                    np.testing.assert_array_equal(oracle, common[t][1])
                else:
                    common[t] = (x.copy(), oracle.copy())
                u, a = x @ Q, 1-t
                c = a*a + t*t*var
                q = t*mu*u/c
                expected = (((t*var-a)/c)*u + a*mu/c*np.tanh(q)) @ Q.T
                discrepancy = float(np.abs(expected-oracle).max())
                max_oracle_error = max(max_oracle_error, discrepancy)
                np.testing.assert_allclose(expected, oracle, atol=1e-12, rtol=1e-12)
                r = 2*np.abs(q)
                e = np.exp(-r)
                h = (np.log1p(e) + r*e/(1+e))/np.log(2.)
                assert np.all((h >= 0) & (h <= 1))
                region = np.searchsorted(EDGES[1:-1], h, side='right')
                if t in bins:
                    np.testing.assert_array_equal(region, bins[t])
                bins[t] = region
                residual = (pred-oracle) @ Q
                errors[arm, t] = residual*residual
                ambient_mse = float(np.mean((pred-oracle)**2))
                latent_mse = float(errors[arm, t].mean())
                max_rotation_mse_error = max(max_rotation_mse_error, abs(ambient_mse-latent_mse))
                np.testing.assert_allclose(latent_mse, ambient_mse, atol=1e-12, rtol=1e-12)
                recorded = next(row['field_mse'] for row in summary['rows'] if row['time'] == t)
                np.testing.assert_allclose(latent_mse, recorded, atol=1e-12, rtol=1e-12)
        errors[arm, 'all_times'] = np.concatenate([errors[arm, t] for t in TIMES])
        np.testing.assert_allclose(errors[arm, 'all_times'].mean(), summary['field_mse'], atol=1e-12, rtol=1e-12)
    bins['all_times'] = np.concatenate([bins[t] for t in TIMES])
    rows = []
    for t in (*TIMES, 'all_times'):
        for arm in ARMS:
            err = errors[arm, t]
            for index, name in [(None, 'all'), *enumerate(REGIONS)]:
                mask = np.ones_like(bins[t], dtype=bool) if index is None else bins[t] == index
                n = int(mask.sum())
                contribution = float(err[mask].sum()/err.size)
                row = dict(label=arm, time=t, entropy_bin=name, coordinate_count=n,
                           coordinate_fraction=n/err.size,
                           conditional_mse=float(err[mask].mean()) if n else '',
                           mse_contribution=contribution, error_share=contribution/float(err.mean()),
                           total_mse=float(err.mean()))
                for baseline in ARMS[1:]:
                    gap = float((err-errors[baseline, t])[mask].sum()/err.size)
                    total_gap = float(err.mean()-errors[baseline, t].mean())
                    row[f'gap_vs_{baseline}'] = gap
                    row[f'gap_share_vs_{baseline}'] = gap/total_gap if arm == 'vector_parent' else ''
                rows.append(row)
            selected = rows[-5:]
            assert sum(row['coordinate_count'] for row in selected[1:]) == err.size
            np.testing.assert_allclose(sum(row['mse_contribution'] for row in selected[1:]), err.mean(), atol=1e-12, rtol=1e-12)
    output = DOCS / 'posterior-velocity-regions.csv'
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = dict(state='PASS', protocol_sha256=PROTOCOL_SHA, source_sha256=sha(__file__),
                      source=str(Path(__file__).resolve()), csv_sha256=sha(output), files_sha256=files,
                      arrays=15, anchors_per_panel=512, coordinates_per_anchor=16, rows=len(rows),
                      entropy_edges=EDGES.tolist(), arms=list(ARMS), times=list(TIMES),
                      max_oracle_reconstruction_error=max_oracle_error,
                      max_rotation_mse_error=max_rotation_mse_error,
                      native_field_mse={a: summaries[a]['field_mse'] for a in ARMS},
                      interpretation='Descriptive attribution of saved velocity error, not measured posterior-classification error or causal decomposition; no new model or data evaluation.')
    (DOCS / 'posterior-velocity-regions.provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
