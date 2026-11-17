"""Independently compare saved compiled arrays with the original eager observer."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('input', 'reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    done = read(a.input/'COMPLETE.json')
    assert done['passed'] and done['tensors_methods_rng_unchanged']
    assert done['reference_complete_sha256'] == sha(a.reference/'COMPLETE.json')
    rows, references = read(a.input/'rows.json'), read(a.reference/'rows.json')
    assert len(rows) == done['rows']
    if not done['witness']:
        assert len(rows) == 8
    results = []
    for row in rows:
        ref = next(r for r in references if (r['case'], r['split']) == (row['case'], row['split']))
        actual_file, reference_file = a.input/row['file'], a.reference/ref['file']
        assert sha(actual_file) == row['sha256']
        assert sha(reference_file) == ref['sha256'] == row['reference_sha256']
        with np.load(actual_file) as data:
            actual = {k: data[k].copy() for k in ('primal', 'tangent')}
        with np.load(reference_file) as data:
            expected = {k: data[k+'256'][:row['points']].copy() for k in actual}
        errors = {}
        for name, value in actual.items():
            reference = expected[name]
            assert value.dtype == reference.dtype == np.float64
            assert value.shape == reference.shape and len(value) == row['points']
            error = np.abs(value-reference)
            assert np.all(error <= 1e-10+(1e-9 if name == 'primal' else 1e-8)*np.abs(reference))
            errors[name] = float(error.max())
            assert errors[name] == row[name+'_error']
        assert row['zero_exact'] and row['native_replay_exact']
        results.append(dict(case=row['case'], split=row['split'], points=row['points'], errors=errors))
    counters = read(a.input/'compiler-counters.json')
    assert not counters.get('graph_break') and not counters.get('unimplemented')
    assert counters['stats']['unique_graphs'] > 0
    if done['timings']:
        timings = read(a.input/'timings.json')
        assert timings == done['timings'] and len(timings) == 3
        speed = statistics.median(r['eager_seconds'] for r in timings)/statistics.median(r['compiled_seconds'] for r in timings)
        assert speed == done['median_speedup']
    else:
        speed = None
    result = dict(passed=True, variant=done.get('variant', 'original'), rows=results,
                  median_speedup=speed, unique_compiled_graphs=counters['stats']['unique_graphs'],
                  complete_sha256=sha(a.input/'COMPLETE.json'), source_sha256=sha(Path(__file__)))
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print('INDEPENDENT_COMPILED_OBSERVER_PASS', len(rows), speed, flush=True)


if __name__ == '__main__':
    main()
