"""Complete fixed-first-layer pilot: distribution response and target quality."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry import read, dump, sha

LABELS = {'state_midpoint4':'Shared P', 'palindrome_operator_midpoint4':'Palindrome',
          'alternating_operator_midpoint4':'Alternating', 'free_isru':'Free ISRU',
          'swiglu64':'SwiGLU', 'swiglu192':'SwiGLU'}
TASKS = ('gmm_ring8_2d', 'rotated_product_bimodal_16d')
COLORS = {'identity':'#666666', 'reference':'#984ea3', 'normal':'#2166ac', 'tangential':'#d95f02', 'ray':'#1b9e77'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, out = Path(args.input), Path(args.output)
    terminal = read(root/'terminal-receipts.json')
    assert terminal['passed'] and all(value == 0 for value in terminal['native_exits'].values())
    assert all(value == 0 for value in terminal['independent_exits'].values())
    rows, bindings, maximum = [], [], 0.
    for task in ['ring', 'product']:
        native, verified = root/'formal'/task, root/'formal'/(task+'-verified')
        complete, proof, summary = read(native/'COMPLETE.json'), read(verified/'COMPLETE.json'), read(verified/'summary.json')
        assert complete['passed'] and proof['passed'] and not complete['witness'] and not proof['witness']
        assert complete['cases'] == proof['cases'] == 5 and complete['per_bank'] == 8
        assert sha(verified/'summary.json') == proof['summary_sha256']
        assert summary['input_complete_sha256'] == sha(native/'COMPLETE.json')
        rows += summary['rows']
        maximum = max(maximum, summary['max_candidate_reconstruction_error'])
        bindings.append(dict(task=task, native_complete_sha256=sha(native/'COMPLETE.json'),
            verified_complete_sha256=sha(verified/'COMPLETE.json'), verified_summary_sha256=sha(verified/'summary.json')))
    assert len(rows) == 10 and sum(row['focal'] for row in rows) == 6
    focal = [row for row in rows if row['focal']]
    qualified = sum(c['numerically_qualified'] for row in focal for c in row['comparisons'])
    supported = sum(c['numerically_qualified'] and c['direction_hypothesis_point_support'] for row in focal for c in row['comparisons'])
    point_support = sum(c['direction_hypothesis_point_support'] for row in focal for c in row['comparisons'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'figures').mkdir()
    dump(out/'summary.json', dict(rows=rows, bindings=bindings, numerically_qualified_comparisons=qualified,
        supported_qualified_comparisons=supported, point_support_comparisons=point_support,
        max_candidate_reconstruction_error=maximum))
    shutil.copyfile(root/'terminal-receipts.json', out/'terminal-receipts.json')
    shutil.copyfile(root/'readiness/READY.json', out/'readiness.json')
    table = '| Task | Model | Heun steps | G normal | G tangential | G exact-energy ray | Numerical gates | Direction prediction |\n|---|---|---:|---:|---:|---:|---|---|\n'
    for row in focal:
        for comparison in row['comparisons']:
            values = comparison['fractions_by_level']['512'][0]
            shown = ['unresolved' if value is None else f'{value:.3f}' for value in values]
            table += f"| {'Ring8' if row['dataset']==TASKS[0] else 'Product16'} | {LABELS[row['family']]} | {comparison['heun_steps']} | {' | '.join(shown)} | {'pass' if comparison['numerically_qualified'] else 'unresolved'} | {'supports' if comparison['direction_hypothesis_point_support'] else 'does not support'} |\n"
    (out/'table.md').write_text(table)
    quality = '| Task | Model | Heun | Native SW2 | Full refinement | Normal | Tangential | Exact-energy ray |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for row in rows:
        for outer in (128, 129):
            scores = []
            for mode in ['identity', 'reference', 'normal', 'tangential', 'ray']:
                arm = next((arm for arm in row['arms'] if arm['heun_steps'] == outer and arm['mode'] == mode
                            and arm['inner_steps'] == (0 if mode == 'identity' else 512)), None)
                scores.append(f"{arm['target_sw2']:.3f}" if arm else '—')
            quality += f"| {'Ring8' if row['dataset']==TASKS[0] else 'Product16'} | {LABELS[row['family']]} | {outer} | {' | '.join(scores)} |\n"
    (out/'target-quality-table.md').write_text(quality)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, task, title in zip(axes, TASKS, ['Ring8','Product16']):
        group = [row for row in focal if row['dataset'] == task]
        for m, (mode, label) in enumerate([('normal','Normal'),('tangential','Tangential'),('ray','Exact-energy ray')]):
            x = np.arange(len(group))+(m-1)*.24
            coarse = [row['comparisons'][0]['fractions_by_level']['512'][0][m] for row in group]
            other = [row['comparisons'][1]['fractions_by_level']['512'][0][m] for row in group]
            ax.bar(x, coarse, width=.23, color=COLORS[mode], label=label)
            ax.scatter(x, other, marker='o', facecolors='none', edgecolors='black', s=32, zorder=3,
                       label='Heun129 (circles)' if m == 0 else None)
        ax.axhline(0, color='.4', lw=1)
        ax.set(xticks=range(len(group)), xticklabels=[LABELS[row['family']] for row in group], title=title,
               ylabel='Generated-distribution change reproduced (G)')
        ax.legend(fontsize=8)
    for ext in ['png','pdf']:
        fig.savefig(out/'figures'/('terminal-distribution-response.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, task, title in zip(axes, TASKS, ['Ring8','Product16']):
        group = [row for row in rows if row['dataset'] == task]
        for index, (mode, color) in enumerate(COLORS.items()):
            x, values = [], []
            for i, row in enumerate(group):
                arm = next((arm for arm in row['arms'] if arm['heun_steps'] == 128 and arm['mode'] == mode
                            and arm['inner_steps'] == (0 if mode == 'identity' else 512)), None)
                if arm:
                    x.append(i+(index-2)*.15)
                    values.append(arm['target_sw2'])
            ax.bar(x, values, width=.145, color=color, label=mode.title())
        ax.set(xticks=range(len(group)), xticklabels=[LABELS[row['family']] for row in group], title=title,
               ylabel='SW2 to common target (64 samples; lower is better)')
        ax.tick_params(axis='x', labelrotation=18)
        ax.legend(fontsize=7, ncol=2)
    for ext in ['png','pdf']:
        fig.savefig(out/'figures'/('target-quality-controls.'+ext), dpi=180)
    plt.close(fig)
    (out/'analysis-report.md').write_text(f'''# From hidden direction to the terminal distribution

**{qualified}/12model/grid comparisons meet the registered numerical gates;
{supported}/12both qualify and support the directional prediction.** The raw
point prediction is supported in{point_support}/12comparisons. All cases,
failed gates and both grids are retained.

## Intervention and primary question

Change only blocks.0.ffn.ode1on each actual field evaluation. All other local
layers, weights, time embedding and native Heunsampler stay fixed. This first
layer passed the earlier interpolation qualification in all six focal models;
the failed all-layer surrogate is not reclassified as passing. Compare native,
full-refinement, normal, tangential and exact-energy-ray arms on the same64
sources: first eight from each original bank. Ordinary SwiGLUand FreeISRUare
evaluated on those same sources/targets. No training is performed.

Gcompares sorted projected endpoint distributions to the full-refinement arm:
1-minus quantile residual energy divided by the native-to-reference energy.
It is not source-paired endpoint error or target-quality improvement.

![Terminal distribution response](figures/terminal-distribution-response.png)

Filled bars use Heun128; open circles use129, both with inner512. These are
two distinct finite sampling grids, not proof of continuous outer convergence.
Inspect the complete numerical gates before interpreting any direction contrast.

{table}

## Agreement with the actual target

![Target quality and ordinary controls](figures/target-quality-controls.png)

TargetSW2answers whether refinement or its component improves the target fit.
Ganswers which component reproduces the refinement's distribution change.
They must not be substituted for each other. The64sample values below cannot
be ranked against historical32768sample evaluations.

{quality}

## Numerical checks

Every arm retains both inner256/512and outer128/129. Five field-call batches
per arm retain inputs, first-layer states, two DOP853references, all exact
candidate outputs and downstream fields. Per-point local/output gates, inner
endpoint shifts, bank/pooledGstability and effect-resolution gates are fixed
in the protocol. Identity trajectories and final native roundtrips are exact.
Both native entries and independent verifiers exit0; candidate reconstruction
error is at most{maximum:.3e}. Raw arrays bind all original banks and checkpoints.

## Claim candidates and next decision

- Claim: the listed fixed-first-layer interventions do or do not support the
  registered directional terminal-distribution prediction on this pilot.
  - Evidence: six focal checkpoints, both fixed grids, both inner resolutions,
    all five arms, DOPsnapshot gates and four ordinary model controls.
  - Allowed wording: conditional empirical distribution response on64sources.
  - Forbidden stronger wording: population generation improvement, continuous
    outer-flow convergence, or an advantage exclusive to ODEparameter sharing.
  - Uncertainty: finite-direction empirical distributions, fixed weights and
    a bounded source pilot. Retain unresolved comparisons explicitly.
  - Next check: decide a full original-bank comparison from all numerical and
    target-quality outcomes; do not choose an arm by the best pilotSW2alone.
  - Decision: keep the measured outcome and its exact scope.
''')
    stats = '# Conditional pilot statistics\n\n'
    stats += ('Six focal and four ordinary checkpoints, seed0only. Each uses the same64\n'
              'ordered sources and targets, eight per original bank, with two outer grids.\n'
              'The18arms per focal model are repeated interventions at fixed weights.\n'
              'No independent-training inference, significance test or populationCIis\n'
              'reported. The registered primary claim requires all six focal cases at\n'
              'both grids to qualify numerically and meet G_T>0,G_T>G_N,G_T>G_ray.\n'
              'Mean/std below describe bank variation only. Gcontrasts and pooledSW2\n'
              'are effect magnitudes, not estimates of a training-population benefit.\n\n')
    for row in rows:
        for arm in row['arms']:
            values = np.asarray(arm['bank_target_sw2'])
            stats += f"- {row['dataset']} / {LABELS[row['family']]} / {arm['mode']} / inner{arm['inner_steps']} / Heun{arm['heun_steps']}: bank SW2 {values.mean():.3f} ± {values.std(ddof=1):.3f}; pooled{arm['target_sw2']:.3f}.\n"
    (out/'stats-appendix.md').write_text(stats)
    (out/'figure-catalog.md').write_text('''# Figure catalog

## terminal-distribution-response
Purpose: test whether directional feature corrections reproduce the full
refinement's empirical terminal-distribution change. Data: pooled64source
Gat inner512for all six focal models, normal/tangential/ray. Filled bars use
Heun128and circles129; circles are another sampler, not confidence intervals.
Caption must state the numerical qualification count and that higherGmeans
closer to the refinement distribution, not necessarily closer to the target.
Inspect all unresolved gates and both grids before supporting a mechanism.

## target-quality-controls
Purpose: separate reproduction of a change from improvement in actual target
quality. Data:64source targetSW2at Heun128/inner512, plus unmodified ordinary
models on the same banks. No inferential error bars. Caption must state64
samples and fixed weights; full two-grid values remain in the numeric table.
Notice whether full refinement improves or harms target matching and whether
tangential correction follows it. Do not rank against historical larger banks.
''')
    print('Complete pilot:', qualified, 'qualified comparisons;', supported, 'qualified supports', flush=True)


if __name__ == '__main__':
    main()
