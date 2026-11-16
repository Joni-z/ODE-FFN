"""Evidence bundle for the registered transported-error calibration."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from geometry import read, dump, sha
from analyze import LABEL, COLORS


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ['input', 'output']:
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    root, out = Path(args.input), Path(args.output)
    terminal = read(root/'terminal-receipts.json')
    assert terminal['passed'] and all(c == 0 for c in terminal['native_exits'].values())
    assert all(c == 0 for c in terminal['independent_exits'].values())
    out.mkdir(parents=True, exist_ok=False)
    (out/'figures').mkdir()
    shutil.copyfile(root/'terminal-receipts.json', out/'terminal-receipts.json')
    rows, raw_rows, max_error = [], {}, 0.
    for name in ['ring', 'product']:
        native, verified = root/name, root/(name+'-verified')
        complete, check, result = read(native/'COMPLETE.json'), read(verified/'COMPLETE.json'), read(verified/'summary.json')
        assert complete['passed'] and check['passed'] and not complete['witness'] and not check['witness']
        assert complete['cases'] == check['cases'] == 6 and complete['sources_per_case'] == 256
        assert sha(verified/'summary.json') == check['summary_sha256']
        assert result['input_complete_sha256'] == sha(native/'COMPLETE.json')
        max_error = max(max_error, result['independent_summary_max_error'])
        rows.extend(result['rows'])
        for row in read(native/'rows.json')['rows']:
            raw_rows[row['id']] = (native, row)
        for file, target in [(native/'COMPLETE.json', name+'-native-COMPLETE.json'),
            (native/'rows.json', name+'-native-rows.json'), (verified/'COMPLETE.json', name+'-independent-COMPLETE.json'),
            (verified/'summary.json', name+'-summary.json')]:
            shutil.copyfile(file, out/target)
    assert len(rows) == 12
    qualified = [r for r in rows if r['qualified']]
    useful = [r for r in qualified if r['explained_fraction'][-1] > 0]
    positive_interval = [r for r in qualified if r['uncertainty']['nominal_twelve_contrast_bonferroni'][0] > 0]
    decision = (f'All twelve models are numerically qualified. The full-scale linear prediction beats zero displacement for {len(useful)}/12 point estimates.'
                if len(qualified) == 12 else f'{len(qualified)}/12 models qualify numerically; keep the others unresolved before interpreting the approximation.')
    table = '| Task | Model | Heun steps | Qualified | Displacement MSE | Remainder MSE | Explained fraction | Adjusted conditional interval |\n|---|---|---:|---|---:|---:|---:|---|\n'
    for row in rows:
        lo, hi = row['uncertainty']['nominal_twelve_contrast_bonferroni']
        table += f"| {'Ring8' if row['dataset']=='gmm_ring8_2d' else 'Product16'} | {LABEL[row['family']]} | {row['final_steps']} | {'yes' if row['qualified'] else 'no'} | {row['displacement_mse'][-1]:.3f} | {row['remainder_mse'][-1]:.3f} | {row['explained_fraction'][-1]:.3f} | [{lo:.3f}, {hi:.3f}] |\n"
    (out/'table.md').write_text(table)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    if len(qualified) < 12:
        fig.suptitle('Numerically unqualified rows are grey crosses')
    for axis, dataset, title in zip(axes, ['gmm_ring8_2d', 'rotated_product_bimodal_16d'], ['Ring8', 'Product16']):
        group = [r for r in rows if r['dataset'] == dataset]
        for i, row in enumerate(group):
            lo, hi = row['uncertainty']['nominal_twelve_contrast_bonferroni']
            color = COLORS[LABEL[row['family']]] if row['qualified'] else '.65'
            axis.hlines(i, lo, hi, color=color, lw=2)
            axis.scatter(row['explained_fraction'][-1], i, color=color, s=35,
                         marker='o' if row['qualified'] else 'x')
        axis.axvline(0, color='.4', lw=1)
        axis.set(yticks=range(6), yticklabels=[LABEL[r['family']] for r in group],
                 xlabel='Full-scale explained fraction', title=title)
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('full-scale-calibration.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    for axis, dataset, title in zip(axes, ['gmm_ring8_2d', 'rotated_product_bimodal_16d'], ['Ring8', 'Product16']):
        for row in [r for r in rows if r['dataset'] == dataset]:
            ratio = np.asarray(row['remainder_mse'])/np.maximum(row['displacement_mse'], np.finfo(float).tiny)
            axis.plot([.125, .25, .5, 1.], ratio, 'o-' if row['qualified'] else 'o--',
                      color=COLORS[LABEL[row['family']]], label=LABEL[row['family']], markersize=4)
        axis.axhline(1, color='.4', lw=1)
        axis.set_xscale('log', base=2)
        axis.set(yscale='log', xticks=[.125, .25, .5, 1.], xticklabels=['.125', '.25', '.5', '1'],
                 xlabel='Field-error multiplier lambda', ylabel='Residual / displacement MSE', title=title)
        axis.xaxis.set_minor_locator(NullLocator())
        axis.legend(fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('error-scale-calibration.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    for axis, prefix in zip(axes, ['palindrome', 'alternating']):
        row = next(r for r in rows if r['dataset'] == 'gmm_ring8_2d' and r['family'].startswith(prefix))
        folder, raw = raw_rows[row['id']]
        with np.load(folder/raw['levels'][-1]['file']) as z:
            true, actual = z['oracle_path'][:, 0], z['paths'][-1, :, 0]
            prediction = true+z['tangent_path'][:, 0]
        for value, label, color in [(true, 'Oracle', '#2166ac'), (actual, 'Model', '#d95f02'),
            (prediction, 'Linear prediction', '#1b9e77')]:
            axis.plot(value[:, 0], value[:, 1], 'o-', color=color, label=label, markersize=3)
        axis.scatter(true[0, 0], true[0, 1], color='black', s=30, zorder=5)
        axis.set(title=f"{LABEL[row['family']]} ({'qualified' if row['qualified'] else 'unqualified'})", xlabel='x1', ylabel='x2')
        axis.legend(fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('ring-fixed-source-example.'+ext), dpi=180)
    plt.close(fig)
    (out/'analysis-report.md').write_text(f'''# Transported-error calibration

**{decision}** {len(positive_interval)}/12 have a positive lower adjusted conditional
input interval. This calibrates an approximation, not a new architecture.

## Question and primary quantity

For each existing20kcheckpoint, transport its velocity error along the oracle
path with z'=J_v*z+e. Compare z(1) with the actual model-minus-oracle endpoint
delta from the same source. Explained fraction is1-MSE(delta-z)/MSE(delta):
positive beats predicting no displacement; zero ties that prediction; negative
is worse. All twelve models retain all256sources,32from each of eight fixed
selection banks. Field multipliers.125/.25/.5/1share those sources and weights.
These are synthetic velocity MLPs, not image JiTmodels or training replicates.

{table}

![Full trained error scale](figures/full-scale-calibration.png)

Bars are conditional bank-bootstrap intervals adjusted over the twelve-model
family. Unqualified rows stay visible and do not support an approximation claim.
All original levels, banks and values remain in the bound numerical records.

![Every registered error multiplier](figures/error-scale-calibration.png)

Below the horizontal line, the linear prediction beats no displacement. A good
small-error result cannot replace the full-scale decision. The same input bank
is reused across multipliers, so these points are repeated measurements.

![Fixed Ring8source example](figures/ring-fixed-source-example.png)

This is source0of bank0, chosen by index rather than the observed outcome. It
compares actual accepted states at nine registered times for the two focal
schedules. The black point is the source. It illustrates the approximation;
the table uses every source rather than this one example.

## Numerical and interpretation limits

Refinement starts128/256and stops only after all registered endpoint/tangent
and fraction-change gates pass, or at1024. The all-source nominal fraction
change is retained per model. Independent NumPy reconstruction differs by at
most{max_error:.2e}. Both native tasks and both independent verifiers exit0.
Oracle JVPs, exact Gaussian solutions, native FP32anchors, FP64conversion and
discrete-tangent finite differences pass in the separate readiness bundle.

A positive explained fraction concerns a particular source-to-endpoint coupling.
It is notSW2and does not imply better generation: the Gaussian rotation control
changes paired endpoints while preserving the exact law. The preexisting
large-bank SW2and ordinary controls remain authoritative for generation.

## Decision

Keep all twelve outcomes model specific. If any numerical rows remain unresolved,
resolve them before claiming a failure of the linear theory. For qualified models
with useful full-scale predictions, a separate distribution-level transport test
is permitted by the prior protocol. Other models reject use of this first-order
approximation at their trained error scale. No new architecture or training
follows from a local Taylor expansion alone.

## Claim candidates

- Claim: this sensitivity equation predicts the learned field's endpoint error.
  Evidence: full-scale calibration and its numerical gates. Decision: model specific.
- Claim: smaller paired endpoint error guarantees lowerSW2.
  Evidence: no; the rotational-law control separates those objects. Decision: reject.
- Claim: these observations establish an ODE-specific generation advantage.
  Evidence: no matched architectural intervention here. Decision: not established.
''')
    (out/'stats-appendix.md').write_text('''# Units and uncertainty

Twelve fixed checkpoints, eight common input banks per task,32sources per bank.
Jointly resample each bank's numerator and denominator sums before taking the
explained-fraction ratio. Do not average bank ratios or treat times/multipliers
as independent samples. Exact bootstrap enumerates6435multisets weighted to
8^8ordered resamples. Report95% and nominal99.583% intervals for the twelve-model
family. These concern fixed-weight input variation, not independent trainings.
No Gaussian assumption, across-model p-value or training significance is claimed.
Numerically unqualified rows remain visible and are not interpreted as model failures.
''')
    (out/'figure-catalog.md').write_text('''# Figure catalogue

- full-scale-calibration: primary approximation utility for every checkpoint.
  Dots are fractions; bars are exact conditional intervals adjusted across twelve
  models; grey crosses mark unqualified rows. Notice the sign and numerical status.
- error-scale-calibration: all four registered multipliers, same weights/sources.
  Ratio below1means useful prediction. Notice whether a valid local expansion
  remains useful at full learned error; do not select a favorable multiplier.
- ring-fixed-source-example: bank0/source0for both focal schedules, all nine
  recorded times. The example shows geometry of the prediction error; it does
  not replace the complete256-source metrics or establish a distribution claim.
''')
    dump(out/'summary.json', dict(passed=True, rows=rows, qualified_models=len(qualified),
        useful_full_scale_point_estimates=len(useful), positive_adjusted_input_intervals=len(positive_interval),
        independent_max_error=max_error))
    dump(out/'COMPLETE.json', dict(passed=True, cases=12, sources_per_case=256,
        qualified_models=len(qualified), source_sha256=sha(Path(__file__)),
        artifacts={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}))
    print(decision, flush=True)


if __name__ == '__main__':
    main()
