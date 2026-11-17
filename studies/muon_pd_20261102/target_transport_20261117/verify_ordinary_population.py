"""NumPy-only native-metric reconstruction for the retained ordinary controls."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distance(x, target, directions):
    return math.sqrt(float(np.mean([np.mean((np.sort(x@d)-np.sort(target@d))**2)
                                    for d in directions])))


def statistics(x, spec):
    law = spec['distribution']
    if spec['dim'] == 2:
        labels = np.argmax(x@np.asarray(law['means']).T, axis=1)
        counts = np.bincount(labels, minlength=8)
        return dict(counts=counts.tolist(),
                    sector_l1=float(np.abs(counts/len(x)-np.asarray(law['weights'])).sum()))
    latent = x@np.asarray(law['rotation'])
    observed, expected = [], []
    for axis in (0, 7, 15):
        location, scale = law['locations'][axis], math.sqrt(law['variances'][axis])
        phi = lambda z: (1+math.erf(z/math.sqrt(2)))/2
        expected.append(phi(-location/(2*scale))-phi(-3*location/(2*scale)))
        observed.append(float(np.mean(np.abs(latent[:, axis]) < location/2)))
    return dict(gap_axes=[0, 7, 15], gap_mass=observed, gap_expected=expected,
                gap_error_max=float(np.max(np.abs(np.asarray(observed)-expected))))


def main():
    parser = argparse.ArgumentParser()
    for name in ('binding', 'input', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    assert not a.output.exists()
    binding = read(a.binding)
    complete = read(a.input/'COMPLETE.json')
    assert complete['passed'] and complete['parameters_buffers_rng_unchanged']
    assert complete['binding_sha256'] == sha(a.binding)
    entry = binding['cases'][complete['index']]
    assert complete['case'] == entry['case']
    root, case = Path(binding['control_root']), entry['case']
    for key in ('config', 'checkpoint', 'dataset_manifest'):
        assert sha(root/case[key]) == case[key+'_sha256']
    spec = read(root/case['dataset_manifest'])['datasets'][case['dataset']]
    rows = read(a.input/'rows.json')
    assert len(rows) == complete['rows'] == 4
    assert {(r['split'], r['steps']) for r in rows} == {
        (split, steps) for split in ('selection', 'confirmation') for steps in (128, 2048)}
    errors = []

    def check(actual, expected):
        error = np.abs(np.asarray(actual)-np.asarray(expected))
        assert np.all(error <= 1e-12+1e-10*np.abs(expected)), (actual, expected)
        errors.append(float(error.max()))

    for row in rows:
        source = entry['splits'][row['split']]
        assert sha(source['full_file']) == source['full_sha256']
        assert sha(a.input/row['file']) == row['sha256']
        count, banks = (2, 1) if complete['witness'] else (4096, 8)
        assert row['points_per_bank'] == count and row['banks'] == banks
        with np.load(source['full_file']) as z:
            directions = z['directions'].astype(float)
            targets = [z[f'target_{i}'][:count].astype(float) for i in range(banks)]
            oracle = z['field_oracle'].astype(float)
        with np.load(a.input/row['file']) as z:
            values, field = z['samples'].copy(), z['field'].copy()
        assert values.shape == (banks, count, spec['dim'])
        assert values.dtype == field.dtype == np.float64
        assert np.isfinite(values).all() and np.isfinite(field).all()
        check(row['field_mse'], np.mean((field-oracle)**2))
        check(row['bank_sw2'], [distance(x, y, directions) for x, y in zip(values, targets, strict=True)])
        check(row['pooled_sw2'], distance(np.concatenate(values), np.concatenate(targets), directions))
        expected = statistics(np.concatenate(values), spec)
        assert set(expected) == set(row['sample_stats'])
        for key, value in expected.items():
            check(row['sample_stats'][key], value)
        print('VERIFIED_ORDINARY', row['case'], row['split'], row['steps'], flush=True)
    result = dict(passed=True, witness=complete['witness'], rows=4,
        maximum_metric_error=max(errors), complete_sha256=sha(a.input/'COMPLETE.json'),
        rows_sha256=sha(a.input/'rows.json'), binding_sha256=sha(a.binding),
        source_sha256=sha(__file__), independent_backend='NumPy; no model execution')
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('INDEPENDENT_ORDINARY_PASS', complete['index'], max(errors), flush=True)


if __name__ == '__main__':
    main()
