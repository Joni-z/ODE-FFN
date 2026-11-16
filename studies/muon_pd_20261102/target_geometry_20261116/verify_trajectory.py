"""Independent NumPy endpoint, coverage, qualification and uncertainty audit."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measured(z: dict, selected=None) -> dict:
    endpoints = z['endpoints'] if selected is None else z['endpoints'][:, selected]
    oracle = z['oracle_endpoint'] if selected is None else z['oracle_endpoint'][selected]
    tangent = z['tangent'] if selected is None else z['tangent'][selected]
    displacement = endpoints-oracle[None]
    residual = displacement-np.asarray([.125, .25, .5, 1.])[:, None, None]*tangent[None]
    denominator = np.einsum('knd,knd->k', displacement, displacement)
    numerator = np.einsum('knd,knd->k', residual, residual)
    return dict(displacement_mse=(denominator/displacement.shape[1]/displacement.shape[2]).tolist(),
        remainder_mse=(numerator/displacement.shape[1]/displacement.shape[2]).tolist(),
        explained_fraction=(1-numerator/np.maximum(denominator, np.finfo(float).tiny)).tolist())


def intervals(bank_residual: np.ndarray, bank_displacement: np.ndarray) -> dict:
    values, weights = [], []
    assert (bank_displacement > 0).all()
    for indices in itertools.combinations_with_replacement(range(8), 8):
        counts = np.bincount(indices, minlength=8)
        weights.append(math.factorial(8)//math.prod(math.factorial(int(c)) for c in counts))
        values.append(1-float(counts@bank_residual)/float(counts@bank_displacement))
    assert len(values) == 6435 and sum(weights) == 8**8
    order = np.argsort(values)
    cdf = np.cumsum(np.asarray(weights, dtype=np.int64)[order])
    ordered = np.asarray(values)[order]
    at = lambda q: float(ordered[np.searchsorted(cdf, math.ceil(q*8**8))])
    return dict(conditional95=[at(.025), at(.975)],
        nominal_twelve_contrast_bonferroni=[at(.05/24), at(1-.05/24)],
        unit='eight fixed-weight input banks; jointly resample numerator and denominator',
        ordered_bootstrap_resamples=8**8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    complete, data = read(root/'COMPLETE.json'), read(root/'rows.json')
    assert complete['passed'] and complete['cases'] == 6 and complete['banks'] == 8
    assert sha(root/'rows.json') == complete['rows_sha256']
    count = 1 if complete['witness'] else 32
    assert complete['sources_per_case'] == 8*count
    rows, maximum = [], 0.
    for row, identity in zip(data['rows'], data['identities']):
        assert row['id'] == identity['id']
        assert sha(Path(identity['assets_file'])) == identity['assets_sha256']
        with np.load(identity['assets_file']) as original:
            expected_source = np.concatenate([original[f'source_{b}'][:count] for b in range(8)]).astype(float)
        previous, levels = None, []
        for index, level in enumerate(row['levels']):
            assert level['steps'] == [128, 256, 512, 1024][index]
            file = root/level['file']
            assert sha(file) == level['sha256']
            with np.load(file) as archive:
                z = {k: archive[k].copy() for k in archive.files}
            assert all(np.isfinite(a).all() for a in z.values())
            assert np.array_equal(z['source'], expected_source)
            assert np.array_equal(z['bank_index'], np.repeat(np.arange(8), count))
            assert np.array_equal(z['lambdas'], [.125, .25, .5, 1.])
            assert np.array_equal(z['record_times'], np.linspace(0, 1, 9))
            assert np.array_equal(z['oracle_path'][0], expected_source)
            assert np.array_equal(z['tangent_path'][0], np.zeros_like(expected_source))
            assert np.array_equal(z['paths'][:, 0], np.broadcast_to(expected_source, (4, *expected_source.shape)))
            assert np.array_equal(z['paths'][:, -1], z['endpoints'])
            assert np.array_equal(z['oracle_path'][-1], z['oracle_endpoint'])
            assert np.array_equal(z['tangent_path'][-1], z['tangent'])
            summary = measured(z)
            for key, value in summary.items():
                error = float(np.max(np.abs(np.asarray(value)-level[key])))
                maximum = max(maximum, error)
                assert np.allclose(value, level[key], atol=1e-10, rtol=1e-11), (row['id'], key)
            banks = [measured(z, z['bank_index'] == bank) for bank in range(8)]
            for bank, value in zip(level['banks'], banks):
                for key in value:
                    assert np.allclose(bank[key], value[key], atol=1e-10, rtol=1e-11)
            gate = None
            if previous is not None:
                gate = dict(endpoints_allclose=bool(np.all(np.abs(previous['endpoints']-z['endpoints']) <= 1e-4+1e-3*np.abs(z['endpoints']))),
                    oracle_allclose=bool(np.all(np.abs(previous['oracle_endpoint']-z['oracle_endpoint']) <= 1e-4+1e-3*np.abs(z['oracle_endpoint']))),
                    tangent_allclose=bool(np.all(np.abs(previous['tangent']-z['tangent']) <= 1e-4+1e-3*np.abs(z['tangent']))))
                change = abs(measured(previous)['explained_fraction'][-1]-summary['explained_fraction'][-1])
                gate['passed'] = all(gate.values()) and change <= .01
                assert all(bool(level['qualification'][k]) == v for k, v in gate.items())
                assert np.isclose(change, level['qualification']['full_scale_fraction_change'], atol=1e-10, rtol=1e-11)
                assert not gate['passed'] or index == len(row['levels'])-1
            levels.append(dict(steps=level['steps'], **summary, qualification=gate))
            previous = z
        assert row['qualified'] == bool(levels[-1]['qualification']['passed'])
        assert row['qualified'] or levels[-1]['steps'] == 1024
        for key, value in measured(previous).items():
            assert np.allclose(value, row[key], atol=1e-10, rtol=1e-11)
        displacement = previous['endpoints'][-1]-previous['oracle_endpoint']
        residual = displacement-previous['tangent']
        numerator = np.array([np.square(residual[previous['bank_index'] == b]).sum() for b in range(8)])
        denominator = np.array([np.square(displacement[previous['bank_index'] == b]).sum() for b in range(8)])
        uncertainty = intervals(numerator, denominator)
        rows.append(dict(id=row['id'], family=row['family'], dataset=row['dataset'],
            qualified=row['qualified'], final_steps=row['final_steps'], levels=levels,
            **measured(previous), banks=[measured(previous, previous['bank_index'] == b) for b in range(8)],
            bank_residual_sums=numerator.tolist(), bank_displacement_sums=denominator.tolist(),
            uncertainty=uncertainty, final_fraction_change=row['levels'][-1]['qualification']['full_scale_fraction_change']))
    assert len(rows) == 6 and len({r['id'] for r in rows}) == 6
    result = dict(passed=True, witness=complete['witness'], cases=6, rows=rows,
        independent_summary_max_error=maximum, all_source_arrays_exact=True,
        input_complete_sha256=sha(root/'COMPLETE.json'), source_sha256=sha(Path(__file__)))
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (out/'COMPLETE.json').write_text(json.dumps(dict(passed=True, witness=complete['witness'], cases=6,
        summary_sha256=sha(out/'summary.json'), max_error=maximum), indent=2)+'\n')
    print(json.dumps(dict(passed=True, cases=6, max_error=maximum,
        qualified=sum(r['qualified'] for r in rows))), flush=True)


if __name__ == '__main__':
    main()
