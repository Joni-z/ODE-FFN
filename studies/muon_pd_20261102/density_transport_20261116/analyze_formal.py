"""Fixed-checkpoint paired density statistics; hold out confirmation time profiles."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from verify_density import read, sha, npz


def bootstrap_means(values, banks, rng, repetitions=10000):
    """Resample the same path indices for every column, within source banks."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    strata = [np.flatnonzero(banks == b) for b in np.unique(banks)]
    draws = []
    for start in range(0, repetitions, 128):
        size = min(128, repetitions-start)
        indices = np.concatenate([s[rng.integers(0, len(s), (size, len(s)))] for s in strata], 1)
        draws.append(values[indices].mean(1))
    return np.concatenate(draws)


def descriptive(values):
    values = np.asarray(values)
    return dict(n=len(values), mean=float(values.mean()), sd=float(values.std(ddof=1)),
        mc_se=float(values.std(ddof=1)/math.sqrt(len(values))),
        quantiles=np.quantile(values, [0, .01, .5, .99, 1]).tolist(),
        largest_one_percent_contribution_to_mean=float(np.sort(values)[-math.ceil(len(values)*.01):].sum()/len(values)))


def sliced_w2(x, y, directions):
    """NumPy replay of the pinned native equal-count empirical metric."""
    assert len(x) == len(y)
    return float(np.sqrt(np.mean((np.sort(x@directions.T, axis=0)-
                                  np.sort(y@directions.T, axis=0))**2)))


def analyze(verification, data_root, output, reference_verification=None, reference_data=None):
    verification, output = map(Path, [verification, output])
    data_root = Path(data_root)
    coordinate = reference_verification is not None
    assert coordinate == (reference_data is not None)
    receipt, cases = read(verification/'COMPLETE.json'), read(verification/'verified-cases.json')
    count = 2 if coordinate else 8
    assert receipt['passed'] and receipt['cases'] == count and not receipt['witness']
    assert len(cases) == len({c['id'] for c in cases}) == count
    assert all(c['passed'] and not c['witness'] for c in cases)
    assert sha(data_root/'density-data.json') == receipt['density_data_sha256']
    meta = read(data_root/'density-data.json')
    data_bindings = {c['id']: receipt['density_data_sha256'] for c in cases}
    reference_receipt = None
    if coordinate:
        assert {c['family'] for c in cases} == {'contrast_operator_midpoint4'}
        assert {c['dataset'] for c in cases} == {'gmm_ring8_2d', 'rotated_product_bimodal_16d'}
        reference_verification, reference_data = map(Path, [reference_verification, reference_data])
        reference_receipt = read(reference_verification/'COMPLETE.json')
        references = read(reference_verification/'verified-cases.json')
        assert reference_receipt['passed'] and reference_receipt['cases'] == 8 and not reference_receipt['witness']
        assert len(references) == len({c['id'] for c in references}) == 8
        assert all(c['passed'] and not c['witness'] for c in references)
        assert sha(reference_data/'density-data.json') == reference_receipt['density_data_sha256']
        previous_meta = read(reference_data/'density-data.json')
        for name, item in meta['datasets'].items():
            old_item = previous_meta['datasets'][name]
            assert sha(reference_data/old_item['file']) == old_item['sha256'] == item['sha256']
        data_bindings.update({c['id']: reference_receipt['density_data_sha256'] for c in references})
        cases = references+cases
        assert len({c['id'] for c in cases}) == 10
    assets = {}
    for name, item in meta['datasets'].items():
        assert sha(data_root/item['file']) == item['sha256']
        assets[name] = npz(data_root/item['file'])
    rng = np.random.default_rng(0)
    values, summaries = {}, []
    for case in cases:
        folder = Path(case['folder'])
        done = read(folder/'COMPLETE.json')
        assert sha(folder/'COMPLETE.json') == case['completion_sha256']
        assert done['density_data_sha256'] == data_bindings[case['id']]
        for cell in case['cells']:
            file = folder/cell['file']
            assert sha(file) == cell['sha256']
            z = npz(file)
            assert z['log_ratio_source'].shape == (1024,) and z['time_parts'].shape == (1024, 8, 3)
            assert np.array_equal(z['bank'], np.repeat(np.arange(4), 256))
            assert np.array_equal(z['source_index'], np.tile(np.arange(256), 4))
            assert bool(z['qualified'].all()) == cell['qualified']
            assert np.max(np.abs(z['log_ratio_source']+z['time_parts'][:, :, :2].sum((1, 2)))) < 1e-9
            key = case['dataset'], case['family'], cell['mode']
            assert key not in values
            values[key] = z, cell
            target = z['log_ratio_source']
            draws = bootstrap_means(target, z['bank'], rng)
            asset = assets[case['dataset']]
            targets = np.concatenate([asset[f'target_{b}'] for b in range(4)])
            directions = asset['directions']
            summaries.append(dict(id=case['id'], dataset=case['dataset'], family=case['family'], mode=cell['mode'],
                numerical_qualified=cell['qualified'], unresolved=len(cell['unresolved_indices']),
                resolution_counts=cell['resolution_counts'], **descriptive(target),
                bootstrap_95=np.quantile(draws[:, 0], [.025, .975]).tolist(),
                divergence_mean=float(z['log_ratio_divergence'].mean()),
                previous_resolution_mean=float(z['previous_log_ratio_source'].mean()),
                resolution_mean_change=float((target-z['previous_log_ratio_source']).mean()),
                bank_means=[float(target[z['bank'] == b].mean()) for b in range(4)],
                same_sources_sw2_pool=sliced_w2(z['generated'], targets, directions),
                previous_resolution_sw2_pool=sliced_w2(z['previous_generated'], targets, directions),
                same_sources_sw2_banks=[sliced_w2(z['generated'][z['bank'] == b],
                    asset[f'target_{b}'], directions) for b in range(4)]))
    contrasts, discovery = [], None
    for dataset in sorted({k[0] for k in values}):
        # A task's paired intervals do not depend on whether another task has finished.
        contrast_rng = np.random.default_rng(0)
        pairs = [('contrast_operator_midpoint4', f) for f in ['state_midpoint4', 'stage_operator_midpoint4']] if coordinate else [('stage_operator_midpoint4', 'state_midpoint4')]
        for mode, (candidate, reference) in [(m, pair) for m in ['original', 'projected'] for pair in pairs]:
            shared, shared_cell = values[dataset, reference, mode]
            stage, stage_cell = values[dataset, candidate, mode]
            fine = stage['log_ratio_source']-shared['log_ratio_source']
            previous = stage['previous_log_ratio_source']-shared['previous_log_ratio_source']
            divergence = stage['log_ratio_divergence']-shared['log_ratio_divergence']
            draws = bootstrap_means(np.column_stack([fine, previous, divergence]), shared['bank'], contrast_rng)
            qualified = shared_cell['qualified'] and stage_cell['qualified']
            sign_flip = float(fine.mean())*float(previous.mean()) < 0
            family_interval = np.quantile(draws[:, 0], [.003125, .996875] if coordinate else [.00625, .99375]).tolist()
            interval_key = 'bonferroni_eight_contrasts_99_375' if coordinate else 'bonferroni_four_contrasts_98_75'
            comparison = dict(dataset=dataset, mode=mode, direction=('mean-preserving minus '+reference+'; positive favors reference') if coordinate else 'stage minus shared; positive favors sharing',
                numerical_qualified=qualified, resolution_sign_flip=sign_flip,
                valid_for_density_ranking=qualified and not sign_flip, **descriptive(fine),
                bootstrap_95=np.quantile(draws[:, 0], [.025, .975]).tolist(),
                **{interval_key: family_interval},
                previous_resolution_mean=float(previous.mean()),
                resolution_mean_change=float((fine-previous).mean()),
                divergence_mean=float(divergence.mean()),
                bank_means=[float(fine[shared['bank'] == b].mean()) for b in range(4)])
            if coordinate:
                comparison.update(candidate=candidate, reference=reference)
            contrasts.append(comparison)
            if not coordinate and dataset == 'rotated_product_bimodal_16d' and mode == 'projected':
                mask = shared['bank'] == 0
                # Confirmation bank time-bin arrays are deliberately never exported.
                bins = (shared['time_parts'][mask, :, :2]-stage['time_parts'][mask, :, :2]).sum(2)
                assert np.max(np.abs(bins.sum(1)-fine[mask])) < 1e-9
                bin_draws = bootstrap_means(bins, np.zeros(mask.sum(), dtype=int), rng)
                means = bins.mean(0)
                eligible = qualified and not sign_flip and family_interval[0] > 0
                index = int(np.argmax(means)) if eligible else None
                discovery = dict(dataset=dataset, mode=mode, bank=0, independent_sources=int(mask.sum()),
                    mean_stage_minus_shared_by_bin=means.tolist(),
                    paired_95_by_bin=np.quantile(bin_draws, [.025, .975], axis=0).T.tolist(),
                    bin_edges=np.linspace(0, 1, 9).tolist(), intervention_candidate_eligible=eligible,
                    candidate_bin=index, candidate_window=[index/8, (index+1)/8] if eligible else None,
                    confirmation_time_profiles_exported=False)
    assert len(summaries) == (20 if coordinate else 16) and len(contrasts) == (8 if coordinate else 4)
    assert (discovery is None) == coordinate
    output.mkdir(parents=True, exist_ok=False)
    result = dict(passed=True, statistics_only=True, unit='independent common Gaussian initial point at fixed trained weights',
        evaluation_only=True, training_seed=0, analysis_rng_seed=0, bootstrap_repetitions=10000,
        qualified_cells=sum(s['numerical_qualified'] for s in summaries),
        qualified_contrasts=sum(c['valid_for_density_ranking'] for c in contrasts),
        summaries=summaries, contrasts=contrasts, discovery=discovery,
        verification_sha256=sha(verification/'COMPLETE.json'), analysis_source_sha256=sha(__file__))
    if coordinate:
        result.update(coordinate_control=True, reference_verification_sha256=sha(reference_verification/'COMPLETE.json'),
            reference_data_sha256=reference_receipt['density_data_sha256'], confirmation_time_profiles_exported=False)
    (output/'statistics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ['passed', 'qualified_cells', 'qualified_contrasts']}))
    for c in contrasts:
        keys = ['dataset', 'mode', 'mean', 'mc_se', 'valid_for_density_ranking', 'resolution_mean_change']
        keys += ['candidate', 'reference', 'bonferroni_eight_contrasts_99_375'] if coordinate else ['bonferroni_four_contrasts_98_75']
        print(json.dumps({k: c[k] for k in keys}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--verification', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--reference-verification')
    p.add_argument('--reference-data')
    a = p.parse_args()
    analyze(a.verification, a.data, a.output, a.reference_verification, a.reference_data)


if __name__ == '__main__':
    main()
