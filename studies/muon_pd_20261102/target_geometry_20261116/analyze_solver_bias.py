"""Prediction accuracy at trained and asymptotic steps, with full qualification."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from geometry import read, dump, sha

TASKS = ('gmm_ring8_2d', 'rotated_product_bimodal_16d')
LABEL = {'state_midpoint4':'Shared P', 'palindrome_operator_midpoint4':'Palindrome',
         'alternating_operator_midpoint4':'Alternating'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, out = Path(args.input), Path(args.output)
    terminal = read(root/'terminal-receipts.json')
    assert terminal['passed'] and all(value == 0 for value in terminal['native_exits'].values())
    assert all(value == 0 for value in terminal['independent_exits'].values())
    rows, bindings = [], []
    for task in ['ring', 'product']:
        native, verified = root/'formal'/task, root/'formal'/(task+'-verified')
        complete, proof, summary = read(native/'COMPLETE.json'), read(verified/'COMPLETE.json'), read(verified/'summary.json')
        assert complete['passed'] and proof['passed'] and not complete['witness'] and not proof['witness']
        assert complete['cases'] == proof['cases'] == 3 and complete['inputs_per_case'] == 80 and complete['layers_per_case'] == 6
        assert sha(verified/'summary.json') == proof['summary_sha256']
        assert summary['input_complete_sha256'] == sha(native/'COMPLETE.json')
        rows += summary['rows']
        bindings.append(dict(task=task, native_complete_sha256=sha(native/'COMPLETE.json'),
            verified_summary_sha256=sha(verified/'summary.json'), native_seconds=complete['seconds']))
    assert len(rows) == 6
    qualified = sum(row['all_qualified'] for row in rows)
    asymptotic = sum(row['all_qualified'] and row['asymptotic_point_support'] for row in rows)
    total_supported = sum(row['all_qualified'] and row['trained_total_point_support'] for row in rows)
    normal_supported = sum(row['all_qualified'] and row['trained_normal_point_support'] for row in rows)
    adjusted_total = sum(row['all_qualified'] and row['uncertainty']['resolved'] and row['uncertainty']['intervals']['trained_total_fraction']['nominal_twelve_contrast_adjusted'][0] > 0 for row in rows)
    adjusted_normal = sum(row['all_qualified'] and row['uncertainty']['resolved'] and row['uncertainty']['intervals']['trained_normal_fraction']['nominal_twelve_contrast_adjusted'][0] > 0 for row in rows)
    out.mkdir(parents=True, exist_ok=False)
    (out/'figures').mkdir()
    dump(out/'summary.json', dict(rows=rows, bindings=bindings, qualified_cases=qualified,
        asymptotic_supported_cases=asymptotic, trained_total_supported_cases=total_supported,
        trained_normal_supported_cases=normal_supported, adjusted_total_positive_cases=adjusted_total,
        adjusted_normal_positive_cases=adjusted_normal))
    shutil.copyfile(root/'terminal-receipts.json', out/'terminal-receipts.json')
    shutil.copyfile(root/'readiness/READY.json', out/'readiness.json')
    table = '| Task | Model | N4 total B | Adjusted interval | N4 normal B | Adjusted interval | N4 actual enrichment | Predicted enrichment | Qualified |\n|---|---|---:|---|---:|---|---:|---:|---|\n'
    levels_table = '| Task | Model | Steps | Relative total residual | Total B | Normal B | Curvature-only B | Chain-only B |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for row in rows:
        task, model = ('Ring8' if row['dataset'] == TASKS[0] else 'Product16'), LABEL[row['family']]
        trained = row['levels'][0]
        intervals = row['uncertainty']['intervals']
        lo_t, hi_t = intervals['trained_total_fraction']['nominal_twelve_contrast_adjusted']
        lo_n, hi_n = intervals['trained_normal_fraction']['nominal_twelve_contrast_adjusted']
        table += f"| {task} | {model} | {trained['total_fraction']:.3f} | [{lo_t:.3f}, {hi_t:.3f}] | {trained['normal_fraction']:.3f} | [{lo_n:.3f}, {hi_n:.3f}] | {64*trained['actual_normal_share']:.3f} | {64*trained['predicted_normal_share']:.3f} | {'yes' if row['all_qualified'] else 'unresolved'} |\n"
        for level in row['levels']:
            levels_table += f"| {task} | {model} | {level['steps']} | {level['relative_total_residual']:.3e} | {level['total_fraction']:.3f} | {level['normal_fraction']:.3f} | {level['curvature_only_fraction']:.3f} | {level['chain_only_fraction']:.3f} |\n"
    (out/'table.md').write_text(table)
    (out/'all-resolutions.md').write_text(levels_table)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
    colors = ['#2166ac', '#d95f02', '#1b9e77']
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    for ax, task, title in zip(axes, TASKS, ['Ring8', 'Product16']):
        for color, row in zip(colors, [r for r in rows if r['dataset'] == task]):
            steps = [level['steps'] for level in row['levels']]
            values = [level['relative_total_residual'] for level in row['levels']]
            ax.plot(steps, values, '-o', color=color, label=LABEL[row['family']], ms=4)
        ax.set(xscale='log', yscale='log', title=title, xlabel='Native midpoint substeps',
               ylabel='Relative error of the parameter-free bias prediction')
        ax.set_xticks(steps, [str(step) for step in steps])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.legend(fontsize=8)
    for ext in ['png','pdf']:
        fig.savefig(out/'figures'/('bias-prediction-resolution.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    for column, (task, title) in enumerate(zip(TASKS, ['Ring8', 'Product16'])):
        group = [row for row in rows if row['dataset'] == task]
        ax = axes[0, column]
        for i, row in enumerate(group):
            for key, value_key, offset, color, label in [('trained_total_fraction','total_fraction',-.1,'#2166ac','Total error'),
                    ('trained_normal_fraction','normal_fraction',.1,'#d95f02','Potential-normal error')]:
                lo, hi = row['uncertainty']['intervals'][key]['nominal_twelve_contrast_adjusted']
                ax.hlines(i+offset, lo, hi, color=color, lw=2)
                ax.scatter(row['levels'][0][value_key], i+offset, color=color, s=30, label=label if i == 0 else None)
        ax.axvline(0, color='.4', lw=1)
        ax.set(yticks=range(3), yticklabels=[LABEL[row['family']] for row in group], title=title+' / trained4 steps',
               xlabel='Bias prediction fraction B')
        ax = axes[1, column]
        for i, row in enumerate(group):
            for offset, value, marker, color, label in [(-.14,64*row['levels'][0]['actual_normal_share'],'o','#2166ac','Actual4 steps'),
                    (0.,64*row['levels'][-1]['actual_normal_share'],'s','#1b9e77','Actual256 steps'),
                    (.14,64*row['levels'][0]['predicted_normal_share'],'D','#d95f02','Predicted leading bias')]:
                ax.scatter(i+offset, value, color=color, marker=marker, s=36, label=label if i == 0 else None)
        ax.axhline(1, color='.5', lw=1)
        ax.set(xticks=range(3), xticklabels=[LABEL[row['family']] for row in group], title=title+' / normal enrichment',
               ylabel='Normal error share divided by1/64')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    extra_handles, extra_labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles+extra_handles, labels+extra_labels, loc='lower center', bbox_to_anchor=(.5, .005), ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, 1))
    for ext in ['png','pdf']:
        fig.savefig(out/'figures'/('trained-bias-and-normal-enrichment.'+ext), dpi=180)
    plt.close(fig)
    (out/'analysis-report.md').write_text(f'''# Predicting the hidden numerical bias without fitting

**{qualified}/6 model cases qualify numerically. The trained4-step prediction
beats zero in {total_supported}/6 total-error and {normal_supported}/6 normal-error
comparisons; adjusted intervals resolve positivity in {adjusted_total}/6 and
{adjusted_normal}/6 respectively.** The registered fine-step decrease is supported
in {asymptotic}/6 numerically qualified cases.

## Prediction and scope

For the frozen hidden field f=P ISRU(y), the midpoint local defect has leading
term eta^3*g, with g=-H[f,f]/24-J(Jf)/6. Integrating z'=Jz+g along the actual
continuous hidden path predicts endpoint error z(1)/N^2. All coefficients follow
from the method; no regression or fitted scale is used. This is established
modified-equation analysis applied to the recorded trained features.
[Derivation and prior work](../solver-bias-derivation.md).

Keep all six shared/PAL/ALT models, six layers and80 inputs. Reuse native
endpoints at4/8/16/32/64/128/256; no data, checkpoint or layer is selected.
These are the state-updated midpoint models, not the image frozen-D Taylor K8.

![Prediction over step refinement](figures/bias-prediction-resolution.png)

Relative residual compares the complete predicted and measured feature-error
vectors. Fine-step accuracy verifies the asymptotic explanation, but the4-step
point must be examined separately to explain the actual trained computation.

## Trained-scale and canonical-normal prediction

![Trained prediction and normal enrichment](figures/trained-bias-and-normal-enrichment.png)

Top: trained4-step prediction fractions and nominal12-contrast-adjusted input-
bank intervals. Bottom: observed and predicted potential-normal enrichment
relative to the1/64 geometric reference. These are hidden-error predictions,
not output-response or generation-quality fractions.

{table}

The signed curvature and repeated-Jacobian contributions are retained separately.
Their individual prediction fractions appear in [all resolutions](all-resolutions.md);
their RMS magnitudes must not be added because contributions can cancel.

## Independent numerical checks

Every state uses two DOP853 tolerance pairs for the state and variational terms.
A separate state-only integration of f+epsilon*g at both positive and negative
epsilon1e-4/1e-5 estimates the endpoint derivative, without solving the
variational equation. Both perturbation scales must satisfy the registered
componentwise gates. Every unresolved point remains in summary.json. Both
native entries and both independent verifiers have actual exit0.

## Research decision and claim candidates

- Claim: the recorded parameter-free bias prediction succeeds or fails at the
  trained and fine resolutions as listed, on the complete fixed cohort.
  - Evidence: all states/resolutions, source-bound endpoint arrays, two-tolerance
    integration, independent state-perturbation checks and paired bank intervals.
  - Allowed wording: quantitative prediction of hidden midpoint discretization
    error on the qualified cases; distinguish fine and trained resolutions.
  - Forbidden stronger wording: improved targetSW2, a new numerical theorem,
    or an advantage over ordinary FFNs from this diagnostic alone.
  - Uncertainty: fixed learned weights and finite input cohort; this does not
    establish performance under a newly trained solver or in the image model.
  - Next check: compare training with different faithful hidden solvers at equal
    parameter and matrix-work budgets, using ordinary controls. A solver switch
    at frozen weights cannot answer whether faithful continuous dynamics help.
  - Decision: keep the resolved prediction and use it to define that causal test.
''')
    (out/'stats-appendix.md').write_text('''# Units, gates and complete comparisons

Six fixed trained checkpoints,80 original inputs each, six local layers and
seven native resolutions. The prediction has no fitted coefficient. All
states remain in pooled actual/residual energies. Report numerical qualification
separately; unresolved cases cannot support the general claim. The normal
component is projected using q at the qualified original continuous reference.

Primary trained tests are total and normal prediction fraction B>0 at N4,
with12 nominally adjusted contrasts. Eight input banks jointly resample actual
and residual energies; exact6435 weighted multisets represent all8^8 ordered
draws. Ordinary95% and adjusted two-sided intervals are retained; the adjusted
tail is0.05/24. These are fixed-weight input intervals, not training replicates.
The fine-step point prediction requires decreasing relative residual at64,
128and256. All other levels and curvature/chain-only diagnostics are retained.

State references must agree within1e-9 absolute/relative; variational components
within1e-8. Independent central flow perturbations at both epsilon scales must
agree componentwise within2e-6 absolute/2e-5 relative. Detailed per-point arrays,
qualification counts and every source hash remain in native/verifier artifacts.
No FIDorSW2inference is made from this hidden-error fit.
''')
    (out/'figure-catalog.md').write_text('''# Figure catalog

## bias-prediction-resolution
Purpose: test the parameter-free endpoint prediction across resolution, including
the actual trained4-step map. Data: pooled relative residual for all six cases
and all seven levels. Log axes show convergence; no fitted trend line or
training-replicate error bars. Caption must identify qualified cases and
distinguish fine asymptotics from trained-scale explanatory success.

## trained-bias-and-normal-enrichment
Purpose: test trained-scale accuracy and whether the formula predicts canonical
normal concentration. Top shows total/normal Bwith adjusted conditional bank
intervals; zero is the no-prediction baseline. Bottom compares actual4-step,
actual256-step and predicted normal share relative to1/64. These are different
quantities from generation quality. Inspect both trained fit and numerical
qualification before claiming an explanatory mechanism.
''')
    print('Complete bias:', qualified, 'qualified;', total_supported, normal_supported,
          'trained total/normal supports;', asymptotic, 'fine supports', flush=True)


if __name__ == '__main__':
    main()
