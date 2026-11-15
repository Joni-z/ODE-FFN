"""Describe only the complete, independently verified frozen symmetry cohort."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

FAMILIES = ['state_midpoint4', 'stage_operator_midpoint4', 'free_isru', 'swiglu64', 'swiglu192']
LABELS = {'state_midpoint4': 'Shared P', 'stage_operator_midpoint4': 'Stage P',
          'free_isru': 'Free ISRU', 'swiglu64': 'SwiGLU64', 'swiglu192': 'SwiGLU192'}
TASKS = ['gmm_ring8_2d', 'rotated_product_bimodal_16d']
TASK_LABELS = ['Ring8 / C8 rotations', 'Product16 / selected reflections']
COLORS = ['#245c8c', '#c56035', '#45896a', '#8a659e']


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_effect(values):
    x = np.asarray(values, dtype=float)
    assert x.shape == (4,) and np.isfinite(x).all()
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    interval = float(stats.t.ppf(.975, 3) * sd / 2)
    nonzero = x[x != 0]
    p = float(stats.binomtest(int((nonzero > 0).sum()), len(nonzero)).pvalue) if len(nonzero) else 1.
    return dict(differences=x.tolist(), mean=mean, sd=sd,
                approximate_95ci=[mean-interval, mean+interval], positive_banks=int((x > 0).sum()),
                negative_banks=int((x < 0).sum()), exact_sign_p=p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verified', required=True)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    verified, runtime, out = map(Path, [args.verified, args.runtime, args.output])
    assert read(verified/'COMPLETE.json')['passed'] and read(runtime/'queue-finished.json')['passed']
    cases = read(verified/'verified-cases.json')
    assert len(cases) == 8 and all(c['passed'] and not c['witness'] for c in cases)
    assert all((c['base_samples'], c['banks'], c['field_points']) == (4096, 4, 4096) for c in cases)
    assert not out.exists()
    out.mkdir(parents=True)
    figures = out/'figures'
    figures.mkdir()
    pools = [p for c in cases for p in c['pooled']]
    bank_rows = [r for c in cases for r in c['bank_rows']]
    case_for = {(c['pooled'][0]['dataset'], c['pooled'][0]['family']): c for c in cases}
    rows, contrasts = [], []
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    field_fig, field_axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    balance_fig, balance_axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for task_index, task in enumerate(TASKS):
        families = [f for f in FAMILIES if (task, f) in case_for]
        assert len(families) == 4
        for i, family in enumerate(families):
            case = case_for[task, family]
            original = next(p for p in pools if p['dataset'] == task and p['family'] == family and p['mode'] == 'original')
            projected = next(p for p in pools if p['dataset'] == task and p['family'] == family and p['mode'] == 'projected')
            original_banks = [r for r in bank_rows if r['id'] == case['id'] and r['mode'] == 'original']
            projected_banks = [r for r in bank_rows if r['id'] == case['id'] and r['mode'] == 'projected']
            assert len(original_banks) == len(projected_banks) == 4
            for r in original_banks:
                other = next(v for v in projected_banks if v['bank'] == r['bank'])
                axs[task_index].plot([i-.14, i+.14], [r['sw2'], other['sw2']], color=COLORS[i], alpha=.28, lw=.9)
            axs[task_index].scatter([i-.14, i+.14], [original['pooled_sw2'], projected['pooled_sw2']],
                                     c=[COLORS[i], COLORS[i]], marker='D', s=35, zorder=3)
            field = case['field']
            field_axs[task_index].bar(i, field['projected'], color=COLORS[i], label='Preserved component' if i == 0 else None)
            field_axs[task_index].bar(i, field['breaking'], bottom=field['projected'], color=COLORS[i],
                                      alpha=.35, hatch='///', label='Removed symmetry component' if i == 0 else None)
            balance_axs[task_index].plot([i-.14, i+.14], [original['balance_l1'], projected['balance_l1']],
                                         '-o', color=COLORS[i])
            paired = [r['sw2']-next(v['sw2'] for v in projected_banks if v['bank'] == r['bank']) for r in original_banks]
            rows.append(dict(dataset=task, family=family, id=case['id'],
                             total_field_mse=field['total'], projected_field_mse=field['projected'],
                             breaking_field_mse=field['breaking'],
                             original_pooled_sw2=original['pooled_sw2'], projected_pooled_sw2=projected['pooled_sw2'],
                             original_bank_mean=original['bank_mean'], original_bank_sd=original['bank_sd'],
                             projected_bank_mean=projected['bank_mean'], projected_bank_sd=projected['bank_sd'],
                             original_balance_l1=original['balance_l1'], projected_balance_l1=projected['balance_l1'],
                             original_gap_error_max=original.get('gap_error_max'), projected_gap_error_max=projected.get('gap_error_max'),
                             projection_bank_effect=paired_effect(paired)))
        for ax in [axs[task_index], field_axs[task_index], balance_axs[task_index]]:
            ax.set_xticks(range(4), [LABELS[f] for f in families], rotation=12)
            ax.set_title(TASK_LABELS[task_index])
        axs[task_index].set_ylabel('Group-stratified SW2 (lower is better)')
        axs[task_index].text(.02, .98, 'Within model: left original, right projected\nDiamonds: pooled; lines: four paired banks',
                              transform=axs[task_index].transAxes, va='top', fontsize=8)
        axs[task_index].margins(y=.3)
        field_axs[task_index].set_ylabel('Componentwise field MSE on complete orbits')
        balance_axs[task_index].set_ylabel('Eight-bin mass L1 distance from uniform')
        for mode in ['original', 'projected']:
            shared = sorted([r for r in bank_rows if r['dataset'] == task and r['family'] == 'state_midpoint4' and r['mode'] == mode], key=lambda r: r['bank'])
            stage = sorted([r for r in bank_rows if r['dataset'] == task and r['family'] == 'stage_operator_midpoint4' and r['mode'] == mode], key=lambda r: r['bank'])
            delta = [b['sw2']-a['sw2'] for a, b in zip(shared, stage)]
            shared_pool = next(p for p in pools if p['dataset'] == task and p['family'] == 'state_midpoint4' and p['mode'] == mode)
            stage_pool = next(p for p in pools if p['dataset'] == task and p['family'] == 'stage_operator_midpoint4' and p['mode'] == mode)
            contrasts.append(dict(dataset=task, mode=mode, comparison='stage minus shared',
                                  pooled_difference=stage_pool['pooled_sw2']-shared_pool['pooled_sw2'], **paired_effect(delta)))
    field_axs[0].legend(fontsize=8)
    for figure, name in [(fig, 'generation'), (field_fig, 'field-decomposition'), (balance_fig, 'partition-balance')]:
        figure.savefig(figures/(name+'.png'))
        figure.savefig(figures/(name+'.pdf'))
        plt.close(figure)
    # Same16base sources and time grid for each path, already hash-verified above.
    fig, axs = plt.subplots(2, 4, figsize=(12, 6.2), constrained_layout=True)
    for i, family in enumerate([f for f in FAMILIES if (TASKS[0], f) in case_for]):
        case = case_for[TASKS[0], family]
        for j, mode in enumerate(['original', 'projected']):
            path = np.load(runtime/'cases'/case['id']/(mode+'-paths.npy'))
            assert path.shape == (129, 16, 2)
            for k in range(16):
                axs[j, i].plot(path[:, k, 0], path[:, k, 1], lw=.55, alpha=.55)
            axs[j, i].scatter(path[-1, :, 0], path[-1, :, 1], s=10, c=COLORS[i])
            axs[j, i].set_title(LABELS[family]+' / '+mode, fontsize=9)
            axs[j, i].set_aspect('equal', adjustable='box')
            axs[j, i].set(xlim=(-3.6, 3.6), ylim=(-3.6, 3.6))
    fig.savefig(figures/'ring-paths.png')
    fig.savefig(figures/'ring-paths.pdf')
    plt.close(fig)
    fig, axs = plt.subplots(2, 4, figsize=(12, 6.2), constrained_layout=True)
    for i, family in enumerate([f for f in FAMILIES if (TASKS[0], f) in case_for]):
        case = case_for[TASKS[0], family]
        for j, mode in enumerate(['original', 'projected']):
            filename = runtime/'cases'/case['id']/(mode+'-bank0.npy')
            row = next(r for r in read(runtime/'cases'/case['id']/'COMPLETE.json')['sample_rows']
                       if r['bank'] == 0 and r['mode'] == mode)
            assert digest(filename) == row['generated_sha256']
            cloud = np.load(filename).reshape(8, 4096, 2)[:, :128].reshape(-1, 2)
            axs[j, i].scatter(cloud[:, 0], cloud[:, 1], s=1.6, alpha=.35, c=COLORS[i], rasterized=True)
            axs[j, i].set_title(LABELS[family]+' / '+mode, fontsize=9)
            axs[j, i].set_aspect('equal', adjustable='box')
            axs[j, i].set(xlim=(-3.6, 3.6), ylim=(-3.6, 3.6))
    fig.savefig(figures/'ring-distributions.png')
    fig.savefig(figures/'ring-distributions.pdf')
    plt.close(fig)
    order = np.argsort([c['exact_sign_p'] for c in contrasts])
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(contrasts)-rank)*contrasts[index]['exact_sign_p']))
        contrasts[index]['holm_sign_p_four_sharing_contrasts'] = previous
    payload = dict(passed=True, training_seed=0, evaluation_stream=2026112200, independent_banks=4,
                   independent_base_points_per_bank=4096, correlated_group_points_per_bank=32768,
                   rows=rows, sharing_contrasts=contrasts, verified_receipt_sha256=digest(verified/'COMPLETE.json'))
    (out/'comparison.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    flat = [{k: v for k, v in r.items() if k != 'projection_bank_effect'} for r in rows]
    with (out/'scores.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    table = ['| Task | Model | Original SW2 | Projected SW2 | Total MSE x1,000 | Breaking MSE x1,000 |',
             '|---|---|---:|---:|---:|---:|']
    for r in rows:
        label = TASK_LABELS[TASKS.index(r['dataset'])].split(' / ')[0]
        table.append(f"| {label} | {LABELS[r['family']]} | {r['original_pooled_sw2']:.3f} | {r['projected_pooled_sw2']:.3f} | {1000*r['total_field_mse']:.3f} | {1000*r['breaking_field_mse']:.3f} |")
    (out/'numeric-summary.md').write_text('\n'.join(table)+'\n')
    print(json.dumps(dict(passed=True, rows=len(rows), contrasts=len(contrasts), output=str(out))))


if __name__ == '__main__':
    main()
