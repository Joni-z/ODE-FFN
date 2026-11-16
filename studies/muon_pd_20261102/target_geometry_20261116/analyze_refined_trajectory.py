"""Read the completed refined test; retain every model and numerical outcome."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from geometry import read, dump, sha
from analyze import LABEL, COLORS, TASKS


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ['input', 'output']:
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    root, out = Path(a.input), Path(a.output)
    complete, verified = read(root/'verified/COMPLETE.json'), read(root/'verified/summary.json')
    queue = read(root/'formal/queue-finished.json')
    assert complete['passed'] and complete['cases'] == 12 and queue['passed']
    assert sha(root/'verified/summary.json') == complete['summary_sha256']
    assert sha(root/'formal/queue-finished.json') == complete['queue_terminal_sha256']
    assert len(queue['completed']) == 12 and not queue['unstarted']
    assert all(r['exit_code'] == 0 for r in queue['completed'])
    assert all(r['actual_exit_code'] == 0 for r in complete['receipts'])
    rows = sorted(verified['rows'], key=lambda r: (r['dataset'], LABEL[r['family']]))
    assert len(rows) == len({r['id'] for r in rows}) == 12
    out.mkdir(parents=True, exist_ok=False)
    (out/'figures').mkdir()
    for source, dest in [(root/'verified/summary.json', 'summary.json'),
                          (root/'verified/COMPLETE.json', 'independent-COMPLETE.json'),
                          (root/'formal/queue-finished.json', 'queue-finished.json'),
                          (root/'formal/queue-start.json', 'queue-start.json'),
                          (root/'readiness/READY.json', 'readiness.json')]:
        shutil.copyfile(source, out/dest)
    qualified = [r for r in rows if r['qualified']]
    positive = [r for r in qualified if r['explained_fraction'][-1] > 0]
    interval_positive = [r for r in qualified if r['uncertainty']['nominal_twelve_contrast_bonferroni'][0] > 0]
    table = '| Task | Model | Steps | Qualified | Displacement MSE | Residual MSE | Explained fraction | Adjusted conditional interval | Change from1024 |\n|---|---|---:|---|---:|---:|---:|---|---:|\n'
    for r in rows:
        lo, hi = r['uncertainty']['nominal_twelve_contrast_bonferroni']
        table += f"| {'Ring8' if r['dataset']==TASKS[0] else 'Product16'} | {LABEL[r['family']]} | {r['final_steps']} | {'yes' if r['qualified'] else 'no'} | {r['displacement_mse'][-1]:.3f} | {r['remainder_mse'][-1]:.3f} | {r['explained_fraction'][-1]:.3f} | [{lo:.3f}, {hi:.3f}] | {r['full_scale_fraction_change_from_1024']:.3f} |\n"
    (out/'table.md').write_text(table)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.1), constrained_layout=True)
    for ax, dataset, title in zip(axes, TASKS, ['Ring8', 'Product16']):
        group = [r for r in rows if r['dataset'] == dataset]
        for i, r in enumerate(group):
            color = COLORS[LABEL[r['family']]] if r['qualified'] else '.6'
            lo, hi = r['uncertainty']['nominal_twelve_contrast_bonferroni']
            ax.hlines(i, lo, hi, color=color, lw=2)
            ax.scatter(r['explained_fraction'][-1], i, color=color, marker='o' if r['qualified'] else 'x', s=30)
        ax.axvline(0, color='.5', lw=1)
        ax.set(title=title, yticks=range(6), yticklabels=[LABEL[r['family']] for r in group],
               xlabel='Full-scale explained fraction (positive beats zero)')
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('refined-full-scale.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, dataset, title in zip(axes, TASKS, ['Ring8', 'Product16']):
        for r in [r for r in rows if r['dataset'] == dataset]:
            ratio = np.asarray(r['remainder_mse'])/np.asarray(r['displacement_mse'])
            ax.plot([.125, .25, .5, 1.], ratio, 'o-' if r['qualified'] else 'x--',
                color=COLORS[LABEL[r['family']]], label=LABEL[r['family']], markersize=4)
        ax.axhline(1, color='.5', lw=1)
        ax.set_xscale('log', base=2)
        ax.set(yscale='log', xticks=[.125, .25, .5, 1.], xticklabels=['.125', '.25', '.5', '1'],
               xlabel='Field-error multiplier lambda', ylabel='Residual / displacement MSE', title=title)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.legend(fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('refined-error-scale.'+ext), dpi=180)
    plt.close(fig)
    task_summary = {}
    for dataset in TASKS:
        group = [r for r in rows if r['dataset'] == dataset]
        task_summary[dataset] = dict(qualified=sum(r['qualified'] for r in group),
            positive=sum(r['qualified'] and r['explained_fraction'][-1] > 0 for r in group),
            positive_adjusted_interval=sum(r['qualified'] and r['uncertainty']['nominal_twelve_contrast_bonferroni'][0] > 0 for r in group),
            fractions={LABEL[r['family']]: r['explained_fraction'][-1] for r in group})
    dump(out/'decision.json', dict(cases=12, qualified=len(qualified), positive_points=len(positive),
        positive_adjusted_intervals=len(interval_positive), tasks=task_summary,
        input_complete_sha256=sha(root/'verified/COMPLETE.json'),
        scope='fixed-weight source-coupled endpoint prediction; not distribution quality or ODE causality'))
    (out/'analysis-report.md').write_text(f'''# Refined error-transport calibration

**{len(qualified)}/12 models meet every numerical gate. At the trained error
scale, the linear endpoint prediction beats zero displacement for
{len(positive)}/12 qualified point estimates; {len(interval_positive)}/12 have
positive lower adjusted conditional input intervals.** Every ordinary control
and unresolved numerical outcome remains included.

## Purpose and method

Test whether propagating the learned velocity error along the exact oracle
trajectory predicts the observed model-minus-oracle endpoint displacement.
This is the necessary calibration before interpreting the propagation as a
generation mechanism. It uses every original256source per20kcheckpoint,
32from each of eight banks, without training or changing hidden integration.

The exact local sensitivity equation is z'=J_v* z+(v_model-v*), along the
oracle path, with z(0)=0. Compare z(1) with the actual endpoint difference
delta. The primary quantity is F=1-MSE(delta-z)/MSE(delta). Positive Fbeats
predicting zero displacement. Fis not an explained fraction of SW2orFID.

{table}

![Full-scale approximation](figures/refined-full-scale.png)

Points are the complete fixed-source results. Bars jointly resample the
residual and displacement energies across eight input banks, using the exact
registered bootstrap and nominal twelve-contrast adjustment. Grey crosses,
if present, remain numerically unresolved and support no approximation verdict.

![All four error multipliers](figures/refined-error-scale.png)

Below the horizontal line, the linear prediction beats zero. These are the
same sources at each multiplier, not independent training repeats. Good
small-error behavior cannot replace the decision at lambda=1.

## Precision and provenance

The original1024cap and its unresolved rows remain untouched. This separately
registered extension reuses those arrays and computes2048/4096, then8192only
when needed. Every endpoint, oracle and tangent component must satisfy
atol1e-4/rtol1e-3; full-scale Fmust change by at most.01between final levels.
Passing a global RMSalone is insufficient. All available fine-grid native
lambda1endpoints and nine-time paths agree exactly with the independent
native-only qualification. Packed execution changes only batching.

All twelve native cases and twelve NumPyverifiers exit0. Independent summary
error is at most{verified['maximum_reconstruction_error']:.2e}. COMPLETEand
bindings retain every source, raw array hash, frozen source and actual exit.

## Decision

Interpret positive and nonpositive outcomes model by model, and interpret
unqualified rows as unresolved. A positive prediction identifies a useful
source-coupled error approximation; it does not identify an ODE-specific gain.
The Gaussian rotation control already shows that paired displacement can
change while the law remains fixed. Original large-bank native SW2and ordinary
controls therefore remain the generation evidence. Any next mechanism test
must connect timing/direction of transported error to distribution-relevant
changes and retain those controls. No new architecture is promoted here.

## Claim candidates

- Claim: first-order transport predicts trained-scale endpoint displacement.
  - Evidence: every qualified model's F, numerical gates and bank intervals.
  - Allowed wording: model-specific positive or nonpositive prediction benefit.
  - Forbidden stronger wording: linear theory explains generation quality across models.
  - Uncertainty: conditional input-bank intervals; source coupling is fixed.
  - Next check: distribution-relevant transport for models where the approximation is useful.
  - Decision: retain model-specific outcomes.
- Claim: ODE structure provides a general generation advantage.
  - Evidence: not established by this fixed-weight observer.
  - Allowed wording: continuous sensitivity is a diagnostic shared by all models.
  - Forbidden stronger wording: positive Fdemonstrates ODE architecture superiority.
  - Uncertainty: ordinary controls can also have positive F.
  - Next check: a separately registered matched causal intervention.
  - Decision: not established.
''')
    stats = '''# Statistical units and checks

One fixed20kseed0checkpoint per model. The comparison unit for uncertainty is
an input bank, with eight banks containing32sources each. Report every model
and every bank; the same sources are reused across multipliers and resolutions.

For each bank, save residual squared-error sum and displacement squared-error
sum. Jointly resample these sums, then form1-ratio; do not average bank ratios.
The exact bootstrap enumerates6435multisets weighted by their multinomial
counts, totaling8^8ordered resamples. Report conditional95%and nominal
twelve-contrast Bonferroni intervals. No normality assumption or t-test is
used. These are input-conditional intervals, not training-variability estimates.

The scientific null is no improvement over predicting zero displacement,
F<=0. Numerical qualification is a separate deterministic all-component gate.
Positive point estimates alone do not establish a positive adjusted interval.
No sampling-rank claim, p-value or architecture winner is inferred here.

| Task/model | Bank | Full-scale fraction | Residual energy | Displacement energy |\n|---|---:|---:|---:|---:|\n'''
    for r in rows:
        for b, bank in enumerate(r['banks']):
            stats += f"| {'Ring8' if r['dataset']==TASKS[0] else 'Product16'} / {LABEL[r['family']]} | {b} | {bank['explained_fraction'][-1]:.3f} | {r['bank_residual_sums'][b]:.3e} | {r['bank_displacement_sums'][b]:.3e} |\n"
    (out/'stats-appendix.md').write_text(stats)
    (out/'figure-catalog.md').write_text('''# Figure catalog

## refined-full-scale.png / .pdf
- Purpose: decide whether the oracle-path linear prediction is useful at actual trained error.
- Data: independently verified full-scale fractions and adjusted conditional bank intervals.
- Caption: all twelve models; positive beats zero; numerical failures are grey crosses;8banks/model,256sources; fixed checkpoints.
- Observation: read each task/model's sign and interval relative to zero.
- Interpretation: permits or rejects use of this approximation for that model.
- Check: do not translate positive Finto lower SW2or an ODE-specific benefit.

## refined-error-scale.png / .pdf
- Purpose: compare approximation validity across all four registered error magnitudes.
- Data: independently reconstructed residual and displacement MSEs from the final level.
- Caption: same source points and weights at every lambda; line at1is zero-prediction performance; unresolved rows dashed.
- Observation: whether useful small-error prediction persists atlambda1.
- Interpretation: local asymptotics alone cannot justify a trained-scale explanation.
- Check: lambda=.125is a finite perturbation, not proof of the lambda->0limit.
''')
    dump(out/'COMPLETE.json', dict(passed=True, cases=12, qualified=len(qualified),
        source_sha256=sha(Path(__file__)), input_complete_sha256=sha(root/'verified/COMPLETE.json'),
        files={str(f.relative_to(out)): sha(f) for f in sorted(out.rglob('*')) if f.is_file()}))


if __name__ == '__main__':
    main()
