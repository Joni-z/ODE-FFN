"""Independent NumPy audit of refined trajectories and the linear account."""
import argparse
from pathlib import Path
import numpy as np
from geometry import read, dump, sha
from verify_trajectory import measured, intervals


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root, out = Path(a.input), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    complete, row = read(root/'COMPLETE.json'), read(root/'rows.json')
    assert complete['passed'] and complete['banks'] == 8 and complete['id'] == row['id']
    assert sha(root/'rows.json') == complete['rows_sha256']
    count = 1 if complete['witness'] else 32
    assert complete['sources'] == 8*count
    assert sha(Path(row['assets_file'])) == row['assets_sha256']
    with np.load(row['assets_file']) as original:
        expected = np.concatenate([original[f'source_{b}'][:count] for b in range(8)]).astype(float)
    native_complete = Path(row['native_qualification_complete'])
    assert sha(native_complete) == row['native_qualification_complete_sha256']
    proof = read(native_complete)
    assert proof['passed'] and proof['cases'] == 12
    native_summary = read(native_complete.with_name('summary.json'))
    assert sha(native_complete.with_name('summary.json')) == proof['summary_sha256']
    native_binding = next(r for r in native_summary['bindings'] if r['id'] == row['id'])
    native_root = Path(native_binding['native_folder'])
    assert sha(native_root/'rows.json') == native_binding['native_rows_sha256']
    native = read(native_root/'rows.json')
    assert native['checkpoint_sha256'] == row['checkpoint_sha256']
    previous, levels, maximum = None, [], 0.
    index = np.concatenate([np.arange(count)+32*b for b in range(8)])
    for i, level in enumerate(row['levels']):
        assert level['steps'] == [1024, 2048, 4096, 8192][i]
        file = root/level['file']
        assert sha(file) == level['sha256']
        with np.load(file) as archive:
            z = {k: archive[k].copy() for k in archive.files}
        assert all(np.isfinite(v).all() for v in z.values())
        assert np.array_equal(z['source'], expected)
        assert np.array_equal(z['bank_index'], np.repeat(np.arange(8), count))
        assert np.array_equal(z['record_times'], np.linspace(0, 1, 9))
        assert np.array_equal(z['lambdas'], [.125, .25, .5, 1.])
        dim = expected.shape[1]
        for key, shape in [('endpoints', (4, 8*count, dim)), ('paths', (4, 9, 8*count, dim)),
                           ('oracle_endpoint', (8*count, dim)), ('tangent', (8*count, dim)),
                           ('oracle_path', (9, 8*count, dim)), ('tangent_path', (9, 8*count, dim))]:
            assert z[key].shape == shape and z[key].dtype == np.float64
        assert np.array_equal(z['paths'][:, 0], np.broadcast_to(expected, (4, *expected.shape)))
        assert np.array_equal(z['paths'][:, -1], z['endpoints'])
        assert np.array_equal(z['oracle_path'][0], expected)
        assert np.array_equal(z['tangent_path'][0], np.zeros_like(expected))
        assert np.array_equal(z['oracle_path'][-1], z['oracle_endpoint'])
        assert np.array_equal(z['tangent_path'][-1], z['tangent'])
        if i == 0:
            original_file = Path(level['original_file'])
            assert sha(original_file) == level['original_sha256']
            with np.load(original_file) as original:
                for key in ['source', 'oracle_endpoint', 'tangent', 'bank_index']:
                    assert np.array_equal(z[key], original[key][index])
                for key in ['oracle_path', 'tangent_path', 'endpoints']:
                    assert np.array_equal(z[key], original[key][:, index])
                assert np.array_equal(z['paths'], original['paths'][:, :, index])
        native_level = next((r for r in native['levels'] if r['steps'] == level['steps']), None)
        if native_level is not None:
            native_file = native_root/native_level['file']
            assert sha(native_file) == native_level['sha256']
            with np.load(native_file) as original:
                assert np.array_equal(z['endpoints'][-1], original['endpoint'][index])
                assert np.array_equal(z['paths'][-1], original['path'][:, index])
        summary = measured(z)
        banks = [measured(z, z['bank_index'] == b) for b in range(8)]
        for key, value in summary.items():
            error = float(np.max(np.abs(np.asarray(value)-level[key])))
            maximum = max(maximum, error)
            assert np.allclose(value, level[key], atol=1e-10, rtol=1e-11)
        for original_bank, bank in zip(level['banks'], banks):
            for key in bank:
                assert np.allclose(original_bank[key], bank[key], atol=1e-10, rtol=1e-11)
        gate = None
        if previous is not None:
            gate = {}
            for key, name in [('endpoints', 'endpoints_allclose'), ('oracle_endpoint', 'oracle_allclose'),
                               ('tangent', 'tangent_allclose')]:
                gate[name] = bool(np.all(np.abs(previous[key]-z[key]) <= 1e-4+1e-3*np.abs(z[key])))
            change = abs(measured(previous)['explained_fraction'][-1]-summary['explained_fraction'][-1])
            gate['full_scale_fraction_change'] = float(change)
            gate['passed'] = all(gate[k] for k in ['endpoints_allclose', 'oracle_allclose', 'tangent_allclose']) and change <= .01
            assert all(level['qualification'][k] == gate[k] for k in ['passed', 'endpoints_allclose', 'oracle_allclose', 'tangent_allclose'])
            assert np.isclose(level['qualification']['full_scale_fraction_change'], change, atol=1e-10, rtol=1e-11)
            assert i < 2 or not gate['passed'] or i == len(row['levels'])-1
        levels.append(dict(steps=level['steps'], **summary, banks=banks, qualification=gate))
        previous = z
    assert row['final_steps'] == int(previous['steps'])
    assert row['qualified'] == bool(levels[-1]['qualification']['passed'])
    assert row['qualified'] or row['final_steps'] == 8192
    assert row['final_steps'] in [4096, 8192]
    difference = previous['endpoints'][-1]-previous['oracle_endpoint']
    residual = difference-previous['tangent']
    numerator = np.asarray([np.square(residual[previous['bank_index'] == b]).sum() for b in range(8)])
    denominator = np.asarray([np.square(difference[previous['bank_index'] == b]).sum() for b in range(8)])
    result = dict(passed=True, witness=complete['witness'], id=row['id'], dataset=row['dataset'], family=row['family'],
        qualified=row['qualified'], final_steps=row['final_steps'], levels=levels,
        **measured(previous), banks=[measured(previous, previous['bank_index'] == b) for b in range(8)],
        bank_residual_sums=numerator.tolist(), bank_displacement_sums=denominator.tolist(),
        uncertainty=intervals(numerator, denominator),
        full_scale_fraction_change_from_1024=float(measured(previous)['explained_fraction'][-1]-levels[0]['explained_fraction'][-1]),
        all_source_arrays_exact=True, max_reconstruction_error=maximum,
        input_complete_sha256=sha(root/'COMPLETE.json'), source_sha256=sha(Path(__file__)))
    dump(out/'summary.json', result)
    dump(out/'COMPLETE.json', dict(passed=True, witness=complete['witness'], id=row['id'],
        summary_sha256=sha(out/'summary.json'), max_error=maximum))
    print(f"{row['id']}: qualified={row['qualified']}, independent error={maximum:.3e}", flush=True)


if __name__ == '__main__':
    main()
