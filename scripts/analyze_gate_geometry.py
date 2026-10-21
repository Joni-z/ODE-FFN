"""Audit all saved panels and write compact evidence for the gate diagnostic."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_gate_geometry import DATASETS, KINDS, TIMES, panel_metrics, strata, write_json
from synthetic.train import sha256


def compare(expected: object, actual: object) -> None:
    if isinstance(expected, dict):
        assert set(expected) == set(actual)
        for key in expected:
            compare(expected[key], actual[key])
    elif isinstance(expected, (float, int)):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    else:
        assert expected == actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status = json.loads((args.input/'status.json').read_text())
    assert status['state'] == 'COMPLETE' and status['completed_cells'] == 100
    provenance = json.loads((args.input/'provenance.json').read_text())
    assert provenance['smoke'] is False and provenance['checkpoint_step'] == 5000
    assert provenance['training_seed'] == provenance['evaluation_seed'] == 0
    rows = json.loads((args.input/'summary.json').read_text())['results']
    expected = {(d, k, p, t) for d in DATASETS for k in KINDS
                for p in ('interpolation', 'trajectory') for t in TIMES}
    assert {(r['dataset'], r['model'], r['panel'], r['time']) for r in rows} == expected
    assert len(rows) == len(expected)
    common, compact, all_strata, tail_cases, hashes = {}, [], [], [], {}
    points, fd_max, anchor_max = 0, 0., 0.
    for row in rows:
        path = args.input/row['samples']
        assert sha256(path) == row['samples_sha256']
        hashes[path.name] = row['samples_sha256']
        with np.load(path, allow_pickle=False) as f:
            arrays = {key:f[key] for key in f.files}
        assert all(np.isfinite(value).all() for value in arrays.values())
        n = len(arrays['x'])
        assert n == (512 if row['panel'] == 'interpolation' else 128)
        points += n
        compare(row['strata'], strata(arrays, row['dataset'] == 'gmm_ring8_2d'))
        if row['model'] != 'ode_original':
            np.testing.assert_array_equal(arrays['jacobian'], arrays['anchored_jacobian'])
        if row['panel'] == 'interpolation':
            key = row['dataset'], row['time']
            if key in common:
                np.testing.assert_array_equal(arrays['x'], common[key])
            else:
                common[key] = arrays['x'].copy()
        for check in row['checks']:
            anchor_max = max(anchor_max, check['anchor_max_error'])
            fd_max = max(fd_max, check.get('native_fd_relative_max_error', 0),
                         check.get('anchored_fd_relative_max_error', 0))
        identity = {key:row[key] for key in ('dataset','model','panel','time')}
        compact.append({**identity, **row['strata']['all']})
        for name, values in row['strata'].items():
            all_strata.append({**identity, 'stratum':name, **values})
        if row['model'] == 'ode_original' and row['panel'] == 'trajectory':
            errors = panel_metrics(arrays)['native_jacobian_error']
            order = np.argsort(errors)[::-1]
            tail_cases.append({**identity, 'top1_share':float(errors[order[0]]/max(errors.sum(), 1e-15)),
                               'top4_share':float(errors[order[:4]].sum()/max(errors.sum(), 1e-15)),
                               'examples':[dict(source_index=int(i), x=arrays['x'][i].tolist(),
                                                jacobian_squared_error=float(errors[i]),
                                                entropy=float(arrays['entropy'][i]),
                                                log_density=float(arrays['log_density'][i]))
                                           for i in order[:4]]})
    assert points == 32000 and anchor_max < 1e-11 and fd_max < 1e-5
    for name, items in [('geometry-panels.csv', compact), ('geometry-strata.csv', all_strata)]:
        fields = list(dict.fromkeys(key for row in items for key in row))
        with (args.output/name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(items)
    write_json(args.output/'geometry-summary.json', dict(results=rows))
    write_json(args.output/'tail-cases.json', tail_cases)
    write_json(args.output/'provenance.json', provenance)
    write_json(args.output/'audit.json', dict(status='PASS', cells=len(rows), rows=points,
               same_interpolation_states_verified=True, zero_correction_controls_verified=True,
               all_panel_hashes_verified=True, all_stratum_metrics_recomputed=True,
               anchor_output_max_error=anchor_max, finite_difference_relative_max_error=fd_max,
               full_run_seconds=status['seconds'], panel_sha256=hashes,
               raw_summary_sha256=sha256(args.input/'summary.json'),
               analysis_script_sha256=sha256(__file__)))
    print(json.dumps(dict(status='PASS', cells=len(rows), points=points,
                         finite_difference_relative_max_error=fd_max)))


if __name__ == '__main__':
    main()
