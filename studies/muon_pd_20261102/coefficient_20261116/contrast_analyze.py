"""Complete coordinate-control statistics and figures; no checkpoint selection."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import shutil
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sharing_analyze import sha, read, dump, csvrows, holm


def compositions(n: int, k: int):
    if k == 1:
        yield (n,)
    else:
        for first in range(n+1):
            for rest in compositions(n-first, k-1):
                yield (first,)+rest


def bootstrap_interval(delta: np.ndarray) -> list:
    """Complete empirical paired-bank bootstrap; coverage remains approximate."""
    n = len(delta)
    counts = list(compositions(n, n))
    multiplicities = [math.factorial(n)//math.prod(math.factorial(x) for x in c) for c in counts]
    assert sum(multiplicities) == n**n
    values = np.asarray(counts)@delta/n
    order = np.argsort(values)
    cdf = np.cumsum(np.asarray(multiplicities, dtype=float)[order]/n**n)
    return [float(values[order[np.searchsorted(cdf, q)]]) for q in [.025, .975]]


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ['study', 'output']:
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    study, out = Path(a.study), Path(a.output)
    source = study/'analysis'
    done, probe, cost = [read(p) for p in [source/'COMPLETE.json', study/'full-coordinate-probe.json', study/'cost.json']]
    assert done['passed'] and done['training_cases'] == 2 and done['reused_reference_cases'] == 8
    assert probe['passed'] and len(probe['rows']) == 4 and cost['passed'] and len(cost['rows']) == 10
    assert read(source/'pairing.json')['passed']
    for name, count in [('verification.json', 20), ('confirmation-verification.json', 10)]:
        v = read(source/name)
        assert v['passed'] and v['cells'] == 10 and v['rows'] == count and v['max_numpy_difference'] < 1e-10
    # Independent small-size enumeration checks the weighted bootstrap implementation.
    small = np.array([-.2, .1, .7])
    ordered = sorted(np.mean(small[list(i)]) for i in itertools.product(range(3), repeat=3))
    expected = [ordered[math.ceil(q*27)-1] for q in [.025, .975]]
    assert np.allclose(bootstrap_interval(small), expected, atol=1e-15)
    assert bootstrap_interval(np.full(8, .25)) == [.25, .25]
    out.mkdir(parents=True, exist_ok=False)
    figures = out/'figures'
    figures.mkdir()
    for name in ['COMPLETE.json', 'pairing.json', 'verification.json', 'confirmation-verification.json',
                 'selection-metrics.csv', 'selection-banks.csv', 'confirmation-metrics.csv', 'confirmation-banks.csv']:
        shutil.copy2(source/name, out/name)
    rows, banks = csvrows(source/'confirmation-metrics.csv'), csvrows(source/'confirmation-banks.csv')
    assert len(rows) == 10 and len(banks) == 80
    costs = {r['id']: r for r in cost['rows']}
    table = []
    for r in rows:
        c = costs[r['id']]
        assert c['checkpoint']['checkpoint_sha256'] == r['checkpoint_sha256']
        table.append(dict(id=r['id'], dataset=r['dataset'], family=c['family'],
            pooled_sw2=float(r['pooled_sw2']), field_mse=float(r['field_mse']),
            bank_mean=float(r['sw2_mean']), bank_sd=float(r['sw2_sd']), parameters=c['parameters'],
            dense_macs=c['dense_macs_per_example'], forward_cpu_ms=1000*c['forward_cpu_seconds']['median'],
            sample64_cpu_seconds=c['sample_cpu_seconds']['median'], checkpoint_sha256=r['checkpoint_sha256']))
    def values(id: str) -> np.ndarray:
        b = sorted([r for r in banks if r['id'] == id], key=lambda x: int(x['bank']))
        assert [int(x['bank']) for x in b] == list(range(8))
        return np.array([float(x['sw2']) for x in b])
    contrasts = []
    for task in ['gmm_ring8_2d', 'rotated_product_bimodal_16d']:
        group = [r for r in table if r['dataset'] == task]
        new = next(r for r in group if r['family'] == 'contrast_operator_midpoint4')
        for family in ['state_midpoint4', 'stage_operator_midpoint4']:
            reference = next(r for r in group if r['family'] == family)
            delta = values(new['id'])-values(reference['id'])
            nonzero = delta[delta != 0]
            sd = float(delta.std(ddof=1))
            contrasts.append(dict(dataset=task, candidate=new['family'], reference=family,
                primary=family == 'state_midpoint4', bank_differences=delta.tolist(),
                mean_difference=float(delta.mean()), sd_difference=sd,
                bootstrap_percentile95=bootstrap_interval(delta), candidate_better_banks=int((delta < 0).sum()),
                pooled_difference=new['pooled_sw2']-reference['pooled_sw2'],
                standardized_mean_difference=float(delta.mean())/sd if sd else None,
                shapiro_p=float(stats.shapiro(delta).pvalue),
                sign_test_p=float(stats.binomtest(int((nonzero < 0).sum()), len(nonzero), .5).pvalue) if len(nonzero) else 1.))
    for row, corrected in zip(contrasts, holm([r['sign_test_p'] for r in contrasts])):
        row['sign_test_holm_four'] = corrected
    probe_summary = []
    for case in probe['rows']:
        modes = {}
        for mode in ['anchored', 'contrast']:
            layers = [r for b in case['batches'] for r in b[mode]['layers']]
            modes[mode] = dict(mean_step_amplification_median=float(np.median([r['mean_step_amplification'] for r in layers])),
                common_gradient_energy_median=float(np.median([r['gradient_common_energy_fraction'] for r in layers])),
                mean_update_stable_rank_median=float(np.median([r['mean_update']['stable_rank'] for r in layers])),
                field_step_rms_mean=float(np.mean([b[mode]['field_step_rms'] for b in case['batches']])))
        probe_summary.append(dict(id=case['id'], dataset=case['dataset'], coordinates=modes,
            function_max_error=max(b['function_error'] for b in case['batches']),
            physical_gradient_max_error=max(b['physical_gradient_error'] for b in case['batches'])))
    c = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)/2
    old_t, new_t = np.column_stack([np.ones(4), np.eye(4)[:, 1:]]), np.column_stack([np.ones(4), c])
    old_m, new_m = old_t@old_t.T, new_t@new_t.T
    assert np.array_equal(new_m, np.eye(4)+.75*np.ones((4, 4)))
    eigen = [np.linalg.eigvalsh(m) for m in [old_m, new_m]]
    geometry = dict(scope='stage-space SGD gradient metric, not learned P spectrum or AdamW preconditioner',
        anchored_metric=old_m.tolist(), contrast_metric=new_m.tolist(), eigenvalues=[v.tolist() for v in eigen],
        condition_numbers=[float(v[-1]/v[0]) for v in eigen])
    result = dict(passed=True, rows=table, contrasts=contrasts, frozen_function_probes=probe_summary,
        stage_space_geometry=geometry, probe_sha256=sha(study/'full-coordinate-probe.json'),
        cost_sha256=sha(study/'cost.json'), source_sha256=sha(__file__))
    dump(out/'comparison.json', result)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    families = ['state_midpoint4', 'stage_operator_midpoint4', 'contrast_operator_midpoint4', 'free_isru', 'swiglu']
    labels = ['Shared P', 'Anchored stages', 'Mean-preserving', 'Free ISRU', 'SwiGLU']
    colors = ['#2166ac', '#d6604d', '#8c510a', '#757575', '#222222']
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), constrained_layout=True)
    for i, task in enumerate(['gmm_ring8_2d', 'rotated_product_bimodal_16d']):
        data = [next(r for r in table if r['dataset'] == task and r['family'].startswith(f)) for f in families]
        for j, metric in enumerate(['field_mse', 'pooled_sw2']):
            y = [r[metric]*(1000 if j == 0 else 1) for r in data]
            axes[i, j].bar(range(5), y, color=colors)
            axes[i, j].set(xticks=range(5), xticklabels=labels, ylabel='Field MSE x 1000' if j == 0 else 'Pooled SW2',
                title='Ring8' if i == 0 else 'Product16')
            axes[i, j].tick_params(axis='x', rotation=23)
    for ext in ['png', 'pdf']:
        fig.savefig(figures/('coordinate-comparison.'+ext), dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for i, row in enumerate(contrasts):
        lo, hi = row['bootstrap_percentile95']
        axes[0].hlines(i, lo, hi, color='#2166ac' if row['primary'] else '#d6604d', lw=2)
        axes[0].scatter(row['bank_differences'], i+np.linspace(-.09, .09, 8), color='.65', s=15)
        axes[0].scatter(row['mean_difference'], i, color='black', s=28, zorder=3)
    names = [('Ring8' if r['dataset'] == 'gmm_ring8_2d' else 'Product16')+' / new minus '+
        ('shared' if r['primary'] else 'anchored') for r in contrasts]
    axes[0].axvline(0, color='.4', lw=.8)
    axes[0].set(yticks=range(4), yticklabels=names, xlabel='Paired bank SW2 difference; negative favors new', title='Eight common banks; percentile 95% intervals')
    for i, r in enumerate(probe_summary):
        for mode, offset, color in [('anchored', -.12, '#d6604d'), ('contrast', .12, '#8c510a')]:
            axes[1].scatter(i+offset, r['coordinates'][mode]['mean_step_amplification_median'], color=color, s=45,
                label=mode+' coordinates' if i == 0 else None)
    axes[1].axhline(1, color='.4', lw=.8)
    axes[1].set(xticks=range(4), xticklabels=['Ring/old fn', 'Prod/old fn', 'Ring/new fn', 'Prod/new fn'],
        ylabel='Effective mean step / common-P step', title='Same frozen functions, fresh native AdamW state')
    axes[1].tick_params(axis='x', rotation=20)
    axes[1].legend(frameon=False, fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(figures/('paired-banks-and-coordinate-effect.'+ext), dpi=180, bbox_inches='tight')
    plt.close(fig)
    dump(out/'analysis-verification.json', dict(passed=True, models=10, primary_checkpoint=20000,
        paired_banks=8, all_bootstrap_compositions=6435, source_sha256=sha(__file__),
        artifacts={str(f.relative_to(out)): sha(f) for f in out.rglob('*') if f.is_file()}))
    print(json.dumps(dict(passed=True, models=10, contrasts=4, output=str(out))))


if __name__ == '__main__':
    main()
