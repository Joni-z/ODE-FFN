"""Build report tables/figures from completed, source-bound results; no models."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MECH = 'docs/research-assessments/2026-11-16-ode-mechanism/'
PROF = 'docs/research-assessments/2026-11-15-professor-execution/'
BINDINGS = {}


def source(name):
    path = ROOT / name
    BINDINGS[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(name):
    return json.loads(source(name).read_text())


def csv_rows(name):
    with source(name).open() as handle:
        return list(csv.DictReader(handle))


def one(rows, **wanted):
    selected = [r for r in rows if all(r[k] == v for k, v in wanted.items())]
    assert len(selected) == 1, (wanted, len(selected))
    return selected[0]


def table(columns, header, rows, caption, label):
    return ('\\begin{table}[H]\n\\centering\\small\n'
            + '\\begin{tabular}{@{}' + columns + '@{}}\n\\toprule\n'
            + ' & '.join(header) + '\\\\\n\\midrule\n'
            + '\n'.join(' & '.join(row) + '\\\\' for row in rows)
            + '\n\\bottomrule\n\\end{tabular}\n\\caption{' + caption + '}\n'
            + '\\label{' + label + '}\n\\end{table}\n')


def fmt(x):
    return f'{float(x):.3f}'


def main():
    (HERE / 'data').mkdir(exist_ok=True)
    (HERE / 'figures').mkdir(exist_ok=True)
    datasets = ['gmm_ring8_2d', 'rotated_product_bimodal_16d']
    task_names = dict(zip(datasets, ['Ring8', 'Product16']))

    ranks = read(PROF + 'spectra-product16/endpoints.json')
    metrics = csv_rows(PROF + 'results/metrics.csv')
    decomposition = read(PROF + 'results/interpretation-decomposition.json')
    rank_rows = []
    for family, name in [('free_isru', 'Free ISRU'), ('pd_isru_k1', 'ISRU-D')]:
        for optimizer, lr, opt_name in [('adamw', .0003, 'AdamW'), ('muon', .02, 'Muon')]:
            w = one(ranks, family=family, optimizer=optimizer, lr=lr, kind='weight')
            d = one(ranks, family=family, optimizer=optimizer, lr=lr, kind='delta')
            rank_rows.append([name + ' / ' + opt_name, fmt(d['stable_rank_mean']),
                              fmt(w['stable_rank_mean']), fmt(w['entropy_rank_mean']),
                              fmt(w['condition_number_mean'])])
    feedback = table('lrrrr', ['Model / optimizer', r'\shortstack{Update\\stable rank}',
                     r'\shortstack{Weight\\stable rank}', r'\shortstack{Weight\\entropy rank}',
                     r'\shortstack{Weight\\condition no.}'], rank_rows,
                     'Product16 at 20k updates; means over six learned matrices. '
                     'The matrix learning rates are 0.0003 for AdamW and 0.02 '
                     'for Muon; other parameters use AdamW 0.0003.', 'tab:ranks')
    response_rows = []
    for dataset in datasets:
        for family, label in [('pd_isru_k1', 'ISRU'), ('pd_tanh_k1', 'Scaled tanh')]:
            variant = 'native' if family == 'pd_tanh_k1' or dataset == datasets[0] else 'historical_native'
            row = one(metrics, dataset=dataset, family=family, step='20000', variant=variant)
            response_rows.append([task_names[dataset], label, fmt(float(row['field_mse'])*1000),
                                  fmt(row['pooled_sw2'])])
    feedback += table('llrr', ['Task', 'Response', r'Field MSE ($\times10^{-3}$)', 'Pooled SW2'],
                      response_rows, 'Matched K1 response comparison at 20k updates, '
                      'AdamW 0.0003 and Heun128. Scaled tanh matches the slope at zero, '
                      'odd symmetry and saturation of ISRU; parameters, initialization '
                      'and data stream are held fixed. Lower is better.', 'tab:response')
    (HERE / 'data/feedback-tables.tex').write_text(feedback)

    symmetry_rows = []
    for family, label in [('pd_isru_k1', 'ISRU-D'), ('pd_tanh_k1', 'Tanh-D'),
                          ('free_isru', 'Free ISRU'), ('swiglu64', 'SwiGLU64 / Muon')]:
        a = one(metrics, dataset=datasets[0], family=family, step='20000', variant='native')
        b = one(metrics, dataset=datasets[0], family=family, step='20000', variant='c8')
        symmetry_rows.append([label, fmt(a['pooled_sw2']), fmt(b['pooled_sw2']),
                              fmt(a['sector_l1']), fmt(b['sector_l1'])])
    (HERE / 'data/symmetry-table.tex').write_text(table('lrrrr', ['Model',
        r'\shortstack{Native\\SW2}', r'\shortstack{C8\\SW2}',
        r'\shortstack{Native\\sector L1}', r'\shortstack{C8\\sector L1}'], symmetry_rows,
        'Ring8 at 20k, Heun128, eight common 4,096-sample banks pooled. '
        'Sector L1 is deviation from equal mode frequencies. C8 uses eight '
        'rotated predictions per field evaluation. Its benefit occurs in ordinary '
        'and ODE-inspired models. Lower is better.', 'tab:symmetry'))

    sharing = read(MECH + 'coordinate-control/comparison.json')
    sharing_rows = []
    for family, label in [('state_midpoint4', 'Shared P'),
                          ('stage_operator_midpoint4', 'Independent stages'),
                          ('contrast_operator_midpoint4', 'Mean-preserving stage release')]:
        sharing_rows.append([label] + [fmt(one(sharing['rows'], dataset=d, family=family)['pooled_sw2'])
                                     for d in datasets])
    structure = table('lrr', ['Operator design', 'Ring8 SW2', 'Product16 SW2'], sharing_rows,
        '20k, Heun128; all rows use the same banks and eight P applications per '
        'local solve. Shared models have 56,706/58,512 parameters on '
        'Ring8/Product16; released models have 130,434/132,240. '
        'The mean-preserving and original releases have the same function class. '
        'This is a fixed-matrix-work comparison.', 'tab:sharing')
    coef = csv_rows(MECH + 'results/analysis/confirmation-metrics.csv')
    state = csv_rows(MECH + 'results/state-analysis/confirmation-metrics.csv')
    stage = csv_rows(MECH + 'results/stage-analysis/confirmation-metrics.csv')
    coefficients = []
    for family, label, rows in [('screen_isru_k8', 'Fixed Taylor K8', coef),
                                ('free_coeff_k8', 'Free Taylor coefficients', coef),
                                ('state_midpoint4', 'Fixed midpoint stages', state),
                                ('free_stage_midpoint4', 'Free midpoint stages', stage)]:
        values = []
        for dataset in datasets:
            row = one(rows, id=f'{dataset}__{family}__adamw_0.0003', dataset=dataset)
            values.append(fmt(row['pooled_sw2']))
        coefficients.append([label] + values)
    structure += table('lrr', ['Coefficient design', 'Ring8 SW2', 'Product16 SW2'], coefficients,
        '20k confirmation, Heun128. Each free-coefficient model starts with the '
        'same function as its fixed counterpart and retains its P applications. '
        'The free Taylor/stage variants add 42/24 scalar parameters. '
        'This cohort differs from Table~\\ref{tab:sharing}; compare rows within this table.',
        'tab:coefficients')
    (HERE / 'data/structure-tables.tex').write_text(structure)

    solver = read(MECH + 'rk4-results/summary.json')
    assert solver['fidelity_support_both_tasks'] and not solver['quality_support_all_splits_tasks']
    forecast = read(MECH + 'learned-defect-field-results/summary.json')
    assert forecast['primary_field_criterion_passes'] and forecast['fitted_coefficients'] == 0
    ordinary = read(MECH + 'ordinary-population-results/summary.json')
    independent = read(MECH + 'ordinary-population-results/independent.json')
    assert independent['passed'] and independent['source_rows'] == 48
    image = read('docs/celeba128-wide-20261114/continuation-160/completed-matched-results.json')
    practical_rows = []
    for label, match in [('State flow / midpoint', {'label': 'M→M'}),
                         ('Free ISRU / AdamW', {'label': 'Free ISRU'}),
                         ('SwiGLU reference', None)]:
        values = []
        for dataset in datasets:
            lookup = match or {'label': 'SwiGLU64 / Muon' if dataset == datasets[0] else 'SwiGLU192 / AdamW'}
            for split in ['selection', 'confirmation']:
                values.append(fmt(one(ordinary['rows'], dataset=dataset, split=split,
                                      steps=2048, **lookup)['pooled_sw2']))
        practical_rows.append([label] + values)
    practical = table('lrrrr', ['Model', r'\shortstack{Ring8\\selection}',
                     r'\shortstack{Ring8\\confirmation}', r'\shortstack{Product16\\selection}',
                     r'\shortstack{Product16\\confirmation}'], practical_rows,
        'Pooled SW2, Heun2048, eight 4,096-sample banks per stream. State flow '
        'and Free ISRU use AdamW 0.0003; Ring8 SwiGLU64 uses Muon 0.02 and '
        'Product16 SwiGLU192 uses AdamW 0.0003. State/Free dense forward MACs '
        'are 227,648/55,616 on Ring8 and 229,440/57,408 on Product16. '
        'These are matrix-work counts, not total training FLOPs.', 'tab:ordinary')
    image_rows = []
    for epoch in [100, 120, 140, 160]:
        image_rows.append([str(epoch)] + [fmt(one(image['arms'][arm]['FID50k'], epoch=epoch)['value'])
                                         for arm in ['baseline', 'K1', 'K8']])
    practical += table('rrrr', ['Epoch', 'Native SwiGLU', 'Wide ISRU-D K1', 'Wide ISRU-D K8'],
        image_rows, 'CelebA128 standard FID50k. Native has 130,405,632 parameters; '
        'both widened candidates have 130,402,560. The protocol uses EMA 0.9999, '
        'Heun50, CFG0 and the same reference. All three completed 160 epochs. '
        'These are frozen-diagonal image blocks, not the refreshed state-flow model.', 'tab:image')
    (HERE / 'data/practical-tables.tex').write_text(practical)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 10.5, 'axes.labelsize': 10,
                         'legend.fontsize': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#25608b', '#d47626']
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.5), layout='constrained')
    for i, (family, label) in enumerate([('state_midpoint4', 'Midpoint ×4'), ('state_rk4_2', 'RK4 ×2')]):
        xx = np.arange(2) + (i-.5)*.30
        rho = [one(solver['fidelity'], dataset=d, family=family)['rho'] for d in datasets]
        quality = [one(solver['metrics'], dataset=d, family=family, split='confirmation')['pooled_sw2']
                   for d in datasets]
        axes[0].bar(xx, rho, .28, color=colors[i], label=label)
        bars = axes[1].bar(xx, quality, .28, color=colors[i], label=label)
        axes[1].bar_label(bars, labels=[fmt(v) for v in quality], padding=3, fontsize=9)
    for ax in axes:
        ax.set_xticks(range(2), ['Ring8', 'Product16'])
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_yscale('log'); axes[0].set_ylim(1e-7, 1e-4)
    axes[0].set_ylabel('Normalized hidden squared error')
    axes[0].set_title('A. ODE solution accuracy')
    axes[0].legend(loc='upper right', frameon=False)
    axes[1].set_ylim(0, .13); axes[1].set_ylabel('Pooled SW2 (lower is better)')
    axes[1].set_title('B. Generated-distribution quality')
    fig.savefig(HERE/'figures/solver-accuracy-quality.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.7, 2.3), layout='constrained')
    cases = ['ring_midpoint_trained', 'ring_rk4_trained', 'product_midpoint_trained', 'product_rk4_trained']
    labels = ['Ring8 / midpoint', 'Ring8 / RK4', 'Product16 / midpoint', 'Product16 / RK4']
    for i, split in enumerate(['selection', 'confirmation']):
        scores = [one(forecast['rows'], case=c, split=split)['score'] for c in cases]
        yy = np.arange(4) + (i-.5)*.18
        ax.scatter(scores, yy, color=colors[i], marker=['o', 's'][i], s=30,
                   label=split.capitalize(), zorder=3)
    ax.set_yticks(range(4), labels); ax.invert_yaxis()
    ax.set_xlim(.94, 1.002); ax.set_xticks([.94, .96, .98, 1.])
    ax.axvline(1., color='#555555', linestyle=':', lw=1)
    ax.set_xlabel('Prediction score (1 = exact; 0 = no-change predictor)')
    ax.grid(axis='x', alpha=.2)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), frameon=False, ncol=2)
    fig.savefig(HERE/'figures/field-forecast.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)

    for old, new in [('results/isru-samples.png', 'ring8-samples.png'),
                     ('results/isru-trajectories.png', 'ring8-trajectories.png')]:
        shutil.copyfile(source(PROF+old), HERE/'figures'/new)
    additional = ['docs/research-assessments/2026-11-15-response-screen/current-comparison.md',
        MECH+'primary-findings.md', MECH+'coordinate-control/analysis-report.md',
        MECH+'population-resolution-results/analysis-report.md',
        MECH+'learned-defect-field-results/analysis-report.md',
        MECH+'learned-defect-field-protocol.md',
        MECH+'solver-bias-results/analysis-report.md',
        MECH+'mathematical-basis.md',
        MECH+'ordinary-population-results/analysis-report.md',
        MECH+'ordinary-population-results/stats-appendix.md',
        MECH+'rk4-results/analysis-report.md', PROF+'findings.md']
    for name in additional:
        source(name)
    evidence = dict(date='2026-11-17', source_sha256=BINDINGS,
        scope='Completed results only; read-only document/figure preparation, no new model computation',
        metric='pooled SW2 within each named cohort; bank uncertainty kept separate',
        training_seed=0, rank_rows=rank_rows, response_rows=response_rows,
        symmetry_rows=symmetry_rows, orbit_decomposition=decomposition,
        sharing_rows=sharing_rows, coefficient_rows=coefficients,
        solver_fidelity=solver['fidelity'], solver_metrics=solver['metrics'],
        field_forecast=[{k:r[k] for k in ['case', 'split', 'score', 'relative_rms_error',
            'global_alignment', 'actual_oracle_risk_change', 'predicted_oracle_risk_change']}
            for r in forecast['rows']], ordinary_native_rows=practical_rows,
        ordinary_all_rows=ordinary['rows'], image_rows=image_rows,
        full_ordinary_comparison_independent=independent,
        next_two_week_test='Proposed compute-matched continuation, not run')
    (HERE/'data/evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(f'Built tables and figures from {len(BINDINGS)} bound sources; no model execution.')


if __name__ == '__main__':
    main()
