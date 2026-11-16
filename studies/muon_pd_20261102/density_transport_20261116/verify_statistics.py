"""Reconstruct paired density intervals using count weights, not indexed means."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from verify_density import read, sha, npz


def weighted_bootstrap(values: np.ndarray, banks: np.ndarray,
                       rng: np.random.Generator, repetitions: int = 10000) -> np.ndarray:
    """Same declared resampling stream; independent count-weight calculation."""
    groups = [np.flatnonzero(banks == b) for b in np.unique(banks)]
    assert sum(map(len, groups)) == len(values)
    output = []
    for start in range(0, repetitions, 128):
        size = min(128, repetitions-start)
        sums = np.zeros((size, values.shape[1]))
        for group in groups:
            n = len(group)
            positions = rng.integers(0, n, (size, n))
            addresses = (positions+np.arange(size)[:, None]*n).ravel()
            counts = np.bincount(addresses, minlength=size*n).reshape(size, n)
            assert np.array_equal(counts.sum(1), np.full(size, n))
            sums += counts@values[group]
        output.append(sums/len(values))
    return np.concatenate(output)


def verify(statistics: Path, verification: Path, reference: Path, output: Path) -> dict:
    stats = read(statistics)
    assert stats['passed'] and stats['coordinate_control'] and stats['discovery'] is None
    assert stats['bootstrap_repetitions'] == 10000 and stats['analysis_rng_seed'] == 0
    assert stats['qualified_cells'] == 20 and len(stats['summaries']) == 20
    assert len(stats['contrasts']) == 8 and not stats['confirmation_time_profiles_exported']
    assert stats['analysis_source_sha256'] == sha(Path(__file__).with_name('analyze_formal.py'))
    assert stats['verification_sha256'] == sha(verification/'COMPLETE.json')
    assert stats['reference_verification_sha256'] == sha(reference/'COMPLETE.json')
    values = {}
    for root, count in [(verification, 2), (reference, 8)]:
        receipt, cases = read(root/'COMPLETE.json'), read(root/'verified-cases.json')
        assert receipt['passed'] and receipt['cases'] == count and not receipt['witness']
        assert len(cases) == count and len({c['id'] for c in cases}) == count
        for case in cases:
            folder = Path(case['folder'])
            assert case['passed'] and not case['witness']
            assert sha(folder/'COMPLETE.json') == case['completion_sha256']
            assert read(folder/'COMPLETE.json')['density_data_sha256'] == receipt['density_data_sha256']
            for cell in case['cells']:
                assert cell['qualified'] and cell['samples'] == 1024 and not cell['unresolved_indices']
                assert sha(folder/cell['file']) == cell['sha256']
                z = npz(folder/cell['file'])
                assert z['qualified'].all() and z['log_ratio_source'].shape == (1024,)
                assert np.array_equal(z['bank'], np.repeat(np.arange(4), 256))
                assert np.array_equal(z['source_index'], np.tile(np.arange(256), 4))
                key = case['dataset'], case['family'], cell['mode']
                assert key not in values
                values[key] = z
    assert len(values) == 20
    maximum = 0.

    def close(actual, expected) -> None:
        nonlocal maximum
        error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        maximum = max(maximum, error)
        assert error < 1e-10, (error, actual, expected)

    def moments(x: np.ndarray) -> dict:
        n = len(x)
        mean = math.fsum(map(float, x))/n
        sd = math.sqrt(math.fsum((float(v)-mean)**2 for v in x)/(n-1))
        return dict(mean=mean, sd=sd, mc_se=sd/math.sqrt(n))

    for row in stats['summaries']:
        assert row['n'] == 1024 and row['numerical_qualified'] and row['unresolved'] == 0
        z = values[row['dataset'], row['family'], row['mode']]
        for key, value in moments(z['log_ratio_source']).items():
            close(row[key], value)
    records = []
    for dataset in sorted({key[0] for key in values}):
        rng = np.random.default_rng(0)
        for mode in ['original', 'projected']:
            for family in ['state_midpoint4', 'stage_operator_midpoint4']:
                new = values[dataset, 'contrast_operator_midpoint4', mode]
                old = values[dataset, family, mode]
                assert np.array_equal(new['bank'], old['bank'])
                columns = np.column_stack([new[k]-old[k] for k in
                    ['log_ratio_source', 'previous_log_ratio_source', 'log_ratio_divergence']])
                assert np.isfinite(columns).all()
                draws = weighted_bootstrap(columns, new['bank'], rng)
                matches = [c for c in stats['contrasts'] if
                    (c['dataset'], c['mode'], c['reference']) == (dataset, mode, family)]
                assert len(matches) == 1 and matches[0]['candidate'] == 'contrast_operator_midpoint4'
                row = matches[0]
                assert row['n'] == 1024 and row['numerical_qualified']
                for key, value in moments(columns[:, 0]).items():
                    close(row[key], value)
                close(row['previous_resolution_mean'], moments(columns[:, 1])['mean'])
                close(row['divergence_mean'], moments(columns[:, 2])['mean'])
                close(row['resolution_mean_change'], moments(columns[:, 0]-columns[:, 1])['mean'])
                close(row['bootstrap_95'], np.quantile(draws[:, 0], [.025, .975]))
                close(row['bonferroni_eight_contrasts_99_375'], np.quantile(draws[:, 0], [.003125, .996875]))
                close(row['bank_means'], [moments(columns[new['bank'] == b, 0])['mean'] for b in range(4)])
                flip = moments(columns[:, 0])['mean']*moments(columns[:, 1])['mean'] < 0
                assert row['resolution_sign_flip'] == flip
                assert row['valid_for_density_ranking'] == (not flip)
                records.append(dict(dataset=dataset, mode=mode, reference=family))
    assert len(records) == 8
    assert stats['qualified_contrasts'] == sum(c['valid_for_density_ranking'] for c in stats['contrasts'])
    result = dict(passed=True, method='count-weight bootstrap and fsum moments',
        summaries=20, paired_contrasts=8, repetitions=10000,
        max_independent_difference=maximum, statistics_sha256=sha(statistics),
        verification_source_sha256=sha(__file__), confirmation_time_profiles_exported=False)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['statistics', 'verification', 'reference-verification', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.statistics, a.verification, a.reference_verification, a.output)))
