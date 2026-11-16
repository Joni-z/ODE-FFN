"""Registered stage-order contrasts from independently reconstructed common banks."""
import argparse
from pathlib import Path
import shutil
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from contrast_analyze import bootstrap_interval
from sharing_analyze import sha, read, dump, csvrows, holm

TASKS = ['gmm_ring8_2d', 'rotated_product_bimodal_16d']
PAL, ALT = 'palindrome_operator_midpoint4', 'alternating_operator_midpoint4'
SHARED = 'state_midpoint4'


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--study', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    study, out = Path(a.study), Path(a.output)
    source = study/'analysis'
    done, cost, activity, verified = [read(f) for f in [source/'COMPLETE.json',
        study/'cost.json', study/'activity.json', study/'activity-verified.json']]
    assert done['passed'] and done['training_cases'] == 4 and done['reused_reference_cases'] == 8
    assert cost['passed'] and len(cost['rows']) == 12
    assert activity['passed'] and verified['passed'] and verified['models'] == 6
    assert verified['source_probe_sha256'] == sha(study/'activity.json')
    assert read(source/'pairing.json')['passed']
    for name, count in [('verification.json', 24), ('confirmation-verification.json', 12)]:
        check = read(source/name)
        assert check['passed'] and check['cells'] == 12 and check['rows'] == count
        assert check['max_numpy_difference'] < 1e-10
    metrics, banks = csvrows(source/'confirmation-metrics.csv'), csvrows(source/'confirmation-banks.csv')
    assert len(metrics) == 12 and len(banks) == 96
    costs = {r['id']: r for r in cost['rows']}
    table = []
    for row in metrics:
        c = costs[row['id']]
        assert c['checkpoint']['checkpoint_sha256'] == row['checkpoint_sha256']
        table.append(dict(id=row['id'], dataset=row['dataset'], family=c['family'],
            field_mse=float(row['field_mse']), pooled_sw2=float(row['pooled_sw2']),
            bank_mean=float(row['sw2_mean']), bank_sd=float(row['sw2_sd']),
            parameters=c['parameters'], dense_macs=c['dense_macs_per_example'],
            forward_cpu_ms=1000*c['forward_cpu_seconds']['median'],
            sample64_cpu_seconds=c['sample_cpu_seconds']['median'],
            checkpoint_sha256=row['checkpoint_sha256']))

    def values(id: str) -> np.ndarray:
        rows = sorted([r for r in banks if r['id'] == id], key=lambda r: int(r['bank']))
        assert [int(r['bank']) for r in rows] == list(range(8))
        return np.array([float(r['sw2']) for r in rows])

    contrasts = []
    for task in TASKS:
        group = {r['family']: r for r in table if r['dataset'] == task}
        for candidate, reference in [(PAL, ALT), (PAL, SHARED), (ALT, SHARED)]:
            left, right = group[candidate], group[reference]
            delta = values(left['id'])-values(right['id'])
            nz = delta[delta != 0]
            contrasts.append(dict(dataset=task, candidate=candidate, reference=reference,
                primary=reference == ALT, bank_differences=delta.tolist(),
                mean_difference=float(delta.mean()), sd_difference=float(delta.std(ddof=1)),
                marginal_bootstrap95=bootstrap_interval(delta), better_banks=int((delta < 0).sum()),
                pooled_difference=left['pooled_sw2']-right['pooled_sw2'],
                sign_test_p=float(stats.binomtest(int((nz < 0).sum()), len(nz), .5).pvalue) if len(nz) else 1.))
    primary = [r for r in contrasts if r['primary']]
    for row, adjusted in zip(primary, holm([r['sign_test_p'] for r in primary])):
        row['sign_test_holm_two_primary'] = adjusted

    mechanisms = []
    for case in activity['rows']:
        assert case['checkpoint']['checkpoint_sha256'] == costs[case['id']]['checkpoint']['checkpoint_sha256']
        local = case['layers']
        interventions = {r['mode']: r for r in case['interventions']}
        leading = [r['mean_flow_expansion']['leading_coefficient_rms'] for r in local]
        if case['family'] == PAL:
            assert interventions['reverse']['prediction_shift_rms'] == 0.
            assert max(leading) < 1e-10
        mechanisms.append(dict(id=case['id'], dataset=case['dataset'], family=case['family'],
            field_mse=case['field_mse'], frozen_interventions=interventions,
            leading_coefficient_rms_median=float(np.median(leading)),
            mean_map_change_over_increment_median=float(np.median([r['mean_relative_shift'] for r in local])),
            stage_drift_variation_median=float(np.median([r['stage_drift_relative_variation'] for r in local])),
            new_gradient_norm_mean=float(np.mean([r['new_gradient_norm'] for r in case['gradient_probes']]))))
    out.mkdir(parents=True, exist_ok=False)
    figures = out/'figures'
    figures.mkdir()
    for name in ['COMPLETE.json', 'pairing.json', 'verification.json', 'confirmation-verification.json',
                 'selection-metrics.csv', 'selection-banks.csv', 'confirmation-metrics.csv', 'confirmation-banks.csv']:
        shutil.copy2(source/name, out/name)
    result = dict(passed=True, rows=table, contrasts=contrasts, mechanisms=mechanisms,
        primary_checkpoint=20000, scope='fixed trained weights; common sampling-bank uncertainty only',
        numerical_refinement_status='pending',
        cost_sha256=sha(study/'cost.json'), activity_verification_sha256=sha(study/'activity-verified.json'),
        source_sha256=sha(__file__))
    dump(out/'comparison.json', result)
    families = [SHARED, 'stage_operator_midpoint4', PAL, ALT, 'free_isru', 'swiglu']
    labels = ['Shared P', 'Released stages', 'Palindrome', 'Alternating', 'Free ISRU', 'SwiGLU']
    colors = ['#2166ac', '#999999', '#1b9e77', '#d95f02', '#757575', '#222222']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for i, task in enumerate(TASKS):
        group = [next(r for r in table if r['dataset'] == task and r['family'].startswith(f)) for f in families]
        for j, metric in enumerate(['field_mse', 'pooled_sw2']):
            axes[i, j].bar(range(6), [r[metric]*(1000 if j == 0 else 1) for r in group], color=colors)
            axes[i, j].set(xticks=range(6), xticklabels=labels, ylabel='Field MSE x 1000' if j == 0 else 'Pooled SW2', title='Ring8' if i == 0 else 'Product16')
            axes[i, j].tick_params(axis='x', rotation=25)
    for ext in ['png', 'pdf']:
        fig.savefig(figures/('matched-order-comparison.'+ext), dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for i, row in enumerate(contrasts):
        lo, hi = row['marginal_bootstrap95']
        ax.hlines(i, lo, hi, color='#2166ac' if row['primary'] else '.6', lw=2)
        ax.scatter(row['bank_differences'], i+np.linspace(-.08, .08, 8), s=15, color='.6')
        ax.scatter(row['mean_difference'], i, s=30, color='black', zorder=3)
    short = {PAL: 'Pal', ALT: 'Alt', SHARED: 'Shared'}
    names = [('Ring8' if r['dataset'] == TASKS[0] else 'Product16')+' / '+short[r['candidate']]+' minus '+short[r['reference']] for r in contrasts]
    ax.axvline(0, lw=.8, color='.4')
    ax.set(yticks=range(6), yticklabels=names, xlabel='Paired bank SW2 difference; marginal 95% bootstrap interval')
    for ext in ['png', 'pdf']:
        fig.savefig(figures/('paired-order-effects.'+ext), dpi=180)
    plt.close(fig)
    dump(out/'analysis-verification.json', dict(passed=True, models=12, primary_contrasts=2,
        descriptive_shared_comparisons=4, all_bootstrap_compositions=6435,
        artifacts={str(f.relative_to(out)): sha(f) for f in out.rglob('*') if f.is_file()}))
    print('matched order analysis complete:', out)


if __name__ == '__main__':
    main()
