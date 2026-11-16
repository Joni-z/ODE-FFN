"""Finite-state hidden-energy evidence; no generation-ranking inference."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry import read, dump, sha
from analyze import LABEL, COLORS, TASKS, fmt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root, out = Path(a.input), Path(a.output)
    terminal = read(root/'terminal-receipts.json')
    assert terminal['passed'] and all(v == 0 for v in terminal['native_exits'].values())
    assert all(v == 0 for v in terminal['independent_exits'].values())
    rows, bindings, maximum = [], [], 0.
    for name in ['ring', 'product']:
        native, verified = root/'formal'/name, root/'formal'/(name+'-verified')
        complete, proof, result = read(native/'COMPLETE.json'), read(verified/'COMPLETE.json'), read(verified/'summary.json')
        assert complete['passed'] and proof['passed'] and not complete['witness'] and not proof['witness']
        assert complete['cases'] == proof['cases'] == 3 and complete['points_per_layer'] == 80
        assert sha(verified/'summary.json') == proof['summary_sha256']
        assert result['input_complete_sha256'] == sha(native/'COMPLETE.json') and result['all_source_arrays_exact']
        rows += result['rows']
        maximum = max(maximum, result['max_reconstruction_error'])
        bindings.append(dict(dataset=name, native_folder=str(native), native_complete_sha256=sha(native/'COMPLETE.json'),
            verified_folder=str(verified), verified_complete_sha256=sha(verified/'COMPLETE.json'),
            verified_summary_sha256=sha(verified/'summary.json')))
    assert len(rows) == len({r['id'] for r in rows}) == 6
    out.mkdir(parents=True, exist_ok=False)
    (out/'figures').mkdir()
    rows.sort(key=lambda r: (r['dataset'], LABEL[r['family']]))
    dump(out/'summary.json', dict(rows=rows, bindings=bindings, max_reconstruction_error=maximum))
    shutil.copyfile(root/'terminal-receipts.json', out/'terminal-receipts.json')
    shutil.copyfile(root/'readiness/READY.json', out/'readiness.json')
    table = '| Task | Model | Qualified local states | Feature error RMS | Normal share | Tangential share | Energy gap RMS | Balance defect RMS | Work change RMS |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:
        s = r['aggregate'][0]
        table += f"| {'Ring8' if r['dataset']==TASKS[0] else 'Product16'} | {LABEL[r['family']]} | {r['qualified_points']}/{r['total_points']} | {fmt(s['feature_error_rms'])} | {s['normal_squared_error_share']:.3f} | {1-s['normal_squared_error_share']:.3f} | {fmt(s['energy_gap_rms'])} | {fmt(s['balance_defect_rms'])} | {fmt(s['work_change_rms'])} |\n"
    (out/'table.md').write_text(table)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    for ax, task, title in zip(axes, TASKS, ['Ring8', 'Product16']):
        for row in [r for r in rows if r['dataset'] == task]:
            label = LABEL[row['family']]
            ax.plot([s['steps'] for s in row['aggregate']], [s['feature_error_rms'] for s in row['aggregate']],
                    'o-' if row['all_qualified'] else 'x--', color=COLORS[label], label=label)
        ax.set_xscale('log', base=2)
        ax.set(yscale='log', xticks=[4, 8, 16, 32], xticklabels=['4', '8', '16', '32'],
            xlabel='Hidden midpoint substeps', ylabel='Feature error RMS versus tight reference', title=title)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('hidden-map-refinement.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    for ax, task, title in zip(axes, TASKS, ['Ring8', 'Product16']):
        group = [r for r in rows if r['dataset'] == task]
        normal = np.asarray([r['aggregate'][0]['normal_squared_error_share'] for r in group])
        x = np.arange(3)
        ax.bar(x, normal, color='#2166ac', label='Normal to potential surface')
        ax.bar(x, 1-normal, bottom=normal, color='#d9d9d9', label='Tangential')
        ax.axhline(1/64, color='black', linestyle=':', lw=1, label='Isotropic direction reference 1/64')
        ax.set(ylim=(0, 1), xticks=x, xticklabels=[LABEL[r['family']] for r in group],
               ylabel='Share of four-step feature squared error', title=title)
        ax.legend(fontsize=7, loc='upper right')
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('normal-and-tangential-feature-error.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    for ax, task, title in zip(axes, TASKS, ['Ring8', 'Product16']):
        group = [r for r in rows if r['dataset'] == task]
        for j, (key, label, color) in enumerate([
                ('energy_gap_rms', 'Potential gap', '#2166ac'),
                ('balance_defect_rms', 'Discrete balance defect', '#d95f02'),
                ('work_change_rms', 'Symmetric-work change', '#1b9e77')]):
            ax.bar(np.arange(3)+(j-1)*.23, [r['aggregate'][0][key] for r in group], width=.23, color=color, label=label)
        ax.set(yscale='log', xticks=range(3), xticklabels=[LABEL[r['family']] for r in group],
               ylabel='Four-step RMS potential/work discrepancy', title=title)
        ax.legend(fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('hidden-energy-budget.'+ext), dpi=180)
    plt.close(fig)
    shares = [r['aggregate'][0]['normal_squared_error_share'] for r in rows]
    qualified = sum(r['qualified_points'] for r in rows)
    (out/'analysis-report.md').write_text(f'''# Hidden ISRU flow: potential and direction

**{qualified}/2880 captured local states qualify against the two continuous
references. The normal component accounts for {min(shares):.3f}--{max(shares):.3f}
of the four-step feature squared error across the six models.** This directly
measures the learned hidden solver; it does not rank generation performance.

## Purpose and inputs

Investigate the observed change in generation when the trained four-step
hidden computation is refined toward its continuous limit. Shared P, palindrome
and alternating checkpoints are fixed at20k, one per task. Each uses the same
80 registered interpolation inputs: two pairs from every bank at five times.
All six local-solve entrances are retained. No new data, weights or training.

For q(y)=ISRU(y), the potential E has gradient q and exact evolution
dE/ds=q^T sym(P)q. The skew part redirects states without direct instantaneous
potential work. Native midpoint need not satisfy the continuous balance.
[Derivation, settings and hypotheses](../hidden-energy-protocol.md).

{table}

![Hidden map refinement](figures/hidden-map-refinement.png)

The reference is the same hidden field, with its four exact stage boundaries.
Two DOP853 tolerances independently qualify each endpoint and its accumulated
symmetric work. A more accurate local map is not asserted to generate better
samples; the earlier full sampling refinement already separates those objects.

![Normal and tangential errors](figures/normal-and-tangential-feature-error.png)

At the reference endpoint, project the coarse-minus-reference feature vector
onto q and its orthogonal complement. These components sum exactly to that
feature difference. Shares pool squared norms over all captured states/layers;
they are not averages of unstable pointwise ratios. The1/64 line is an isotropic
geometric reference, not an estimated null distribution or significance test.
The normal component measures first-order potential change, not every nonlinear
energy effect. The bounded Taylor remainder is independently checked. The
symmetric part of P can also redirect features, so tangential error is not
attributed exclusively to the skew part.

![Potential/work budget](figures/hidden-energy-budget.png)

Potential gap equals symmetric-work change plus discrete balance defect minus
the tight reference closure, with signed per-state quantities. The displayed
RMS magnitudes need not add: work and balance changes can cancel. A large
balance defect alone therefore need not account for the resulting potential gap.

## Interpretation and next check

Feature error, potential change and downstream velocity change are distinct.
The normal/tangential shares describe the hidden difference, not how strongly
the remaining network responds to it. Any proposed energy explanation must
retain the measured directional component and numerical floor. A subsequent
one-layer-at-a-time replacement can test how each component affects the full
output, but requires its own registered native-map checks. No energy penalty,
new activation, architecture promotion or generation claim follows here.

## Verification

Both native dataset commands and both independent verifiers exit0. Independent
maps/work/geometry reconstruct within{maximum:.2e}; every source/bank/time
binding is exact. All native4/8/16/32 maps replay in NumPy within their registered
tolerance. Both continuous references, midpoint outputs and input arrays remain
in hash-bound raw files. Every unresolved state would remain visible; none is
removed. All original parameters and buffers remain unchanged.

## Claim candidates

- Claim: coarse hidden error can be separated into potential-normal and tangential components.
  - Evidence: all states, exact decomposition and independent scalar-potential bounds.
  - Allowed wording: finite-state component shares and measured numerical balance defects.
  - Forbidden stronger wording: these shares explain the sampling gap or prove a new FFN advantage.
  - Uncertainty: deterministic registered states; reference floors are numerical diagnostics.
  - Next check: registered one-layer native interventions on the complete output.
  - Decision: retain measured components, with no generation attribution.
''')
    stats = '''# Units and numerical uncertainty

Six frozen models,36local layers and80inputs per model, totaling2880local
states. Inputs share banks and times across models and recur across layers;
these are repeated deterministic measurements, not2880independent training runs.
No p-value, performance confidence interval or model winner is inferred.

Reference uncertainty uses per-component tightened-tolerance agreement,
work agreement and exact potential closure under the preregistered gates.
The empirical floor combines reference shift and a floating-point scale;
it is not a rigorous error bound. Both reference endpoints and every input
are retained. The normal share pools squared-error sums across states/layers.
The smooth-potential Taylor remainder is checked for every state and step.

| Task/model | Layer | Reference closure max | Qualified states | Four-step normal share | Four-step feature RMS |\n|---|---|---:|---:|---:|---:|\n'''
    for r in rows:
        for layer in r['layers']:
            s = layer['levels'][0]
            stats += f"| {'Ring8' if r['dataset']==TASKS[0] else 'Product16'} / {LABEL[r['family']]} | {layer['layer']} | {layer['reference_closure_max']:.3e} | {layer['qualified_points']}/{layer['points']} | {s['normal_squared_error_share']:.3f} | {fmt(s['feature_error_rms'])} |\n"
    (out/'stats-appendix.md').write_text(stats)
    (out/'figure-catalog.md').write_text('''# Figure catalog

## hidden-map-refinement.png / .pdf
- Purpose: establish scale and refinement of the trained hidden-map difference.
- Data: complete independently verified six-model aggregates, all36layers and2880states.
- Caption:4/8/16/32midpoint substeps, two checked DOP853references; equal original checkpoint and entrance states.
- Observation: compare the map's actual approach to its continuous reference.
- Interpretation: establishes a computational difference, not its generative utility.

## normal-and-tangential-feature-error.png / .pdf
- Purpose: distinguish potential-normal from tangential feature error.
- Data: pooled squared norms at native4substeps from every state/layer.
- Caption: projection onto q at the reference endpoint;1/64is a geometric isotropic benchmark only.
- Observation: read both shares, retaining the direction component.
- Interpretation: motivates a direct downstream intervention rather than inferring generation from energy.

## hidden-energy-budget.png / .pdf
- Purpose: distinguish changes of path work from numerical balance defects.
- Data: RMSof complete signed per-state energy/work discrepancies.
- Caption: RMSbars are not additive; exact signed closure includes the reference's measured residual.
- Observation: inspect whether work and balance terms are large relative to the resulting gap.
- Interpretation: a defect norm alone does not explain the map or full network.
''')
    dump(out/'COMPLETE.json', dict(passed=True, models=6, local_states=2880, qualified_states=qualified,
        source_sha256=sha(Path(__file__)), files={str(f.relative_to(out)): sha(f) for f in sorted(out.rglob('*')) if f.is_file()}))


if __name__ == '__main__':
    main()
