"""Evidence and actual figures for the native temporal-alias counterexample."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry import read, dump, sha
from analyze import LABEL


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ['input', 'output']:
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    root, out = Path(args.input), Path(args.output)
    complete = read(root/'native/COMPLETE.json')
    proof = read(root/'verified/COMPLETE.json')
    assert complete['passed'] and proof['passed'] and complete['cases'] == proof['cases'] == 12
    assert proof['input_complete_sha256'] == sha(root/'native/COMPLETE.json')
    assert complete['rows_sha256'] == sha(root/'native/rows.json')
    rows = read(root/'native/rows.json')['rows']
    assert len(rows) == 12 and len({r['id'] for r in rows}) == 12
    out.mkdir(parents=True, exist_ok=False)
    (out/'figures').mkdir()
    for src, name in [(root/'native/rows.json', 'rows.json'), (root/'native/COMPLETE.json', 'native-COMPLETE.json'),
                      (root/'verified/COMPLETE.json', 'independent-COMPLETE.json')]:
        shutil.copyfile(src, out/name)
    maxima = {key: max(r[key] for r in rows) for key in ['grid_max_difference', 'endpoint_max_difference',
        'fp32_grid_max_difference', 'fp32_endpoint_max_difference']}
    ratios = [r['off_grid_change_over_fitting_error_rms'] for r in rows]
    table = '| Task | Model | Original J error | Alias J error | Off-grid change / fit RMS | FP64 endpoint max | FP32 endpoint max |\n|---|---|---:|---:|---:|---:|---:|\n'
    for r in rows:
        table += f"| {'Ring8' if r['dataset']=='gmm_ring8_2d' else 'Product16'} | {LABEL[r['family']]} | {r['selection_original_jacobian_error']:.3f} | {r['selection_alias_jacobian_error']:.3f} | {r['off_grid_change_over_fitting_error_rms']:.3f} | {r['endpoint_max_difference']:.3e} | {r['fp32_endpoint_max_difference']:.3e} |\n"
    (out/'table.md').write_text(table)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.7), constrained_layout=True)
    for axis, dataset, title in zip(axes, ['gmm_ring8_2d', 'rotated_product_bimodal_16d'], ['Ring8', 'Product16']):
        group = [r for r in rows if r['dataset'] == dataset]
        for i, r in enumerate(group):
            a, b = r['selection_original_jacobian_error'], r['selection_alias_jacobian_error']
            axis.plot([a, b], [i, i], color='.6', lw=2)
            axis.scatter(a, i, color='black', s=30, label='Original field' if i == 0 else None)
            axis.scatter(b, i, color='#2166ac', marker='s', s=28, label='Sampler-aliased field' if i == 0 else None)
        axis.set(yticks=range(6), yticklabels=[LABEL[r['family']] for r in group],
                 xlabel='Selection full velocity-Jacobian error', title=title)
        axis.legend(fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('same-sampler-different-response.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.7), constrained_layout=True)
    for axis, dataset, title in zip(axes, ['gmm_ring8_2d', 'rotated_product_bimodal_16d'], ['Ring8', 'Product16']):
        group = [r for r in rows if r['dataset'] == dataset]
        for key, label, color in [('endpoint_max_difference', 'FP64', '#2166ac'),
                                  ('fp32_endpoint_max_difference', 'FP32', '#d95f02')]:
            axis.scatter([r[key] for r in group], range(6), label=label, color=color, s=35)
        axis.set(xscale='log', yticks=range(6), yticklabels=[LABEL[r['family']] for r in group],
                 xlabel='Maximum native Heun128 endpoint difference', title=title)
        axis.legend(fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('rounding-level-sample-differences.'+ext), dpi=180)
    plt.close(fig)
    (out/'analysis-report.md').write_text(f'''# Temporal aliasing in the existing synthetic model

**The previously used dyadic sampling checks cannot distinguish two different
continuous fields constructed from each existing checkpoint.** This is a
measured numerical blind spot, not a new architecture or a demonstration that
all earlier sampling differences were caused by aliasing.

## Purpose and exact construction

The native time embedding includes frequency1000.000122cycles/unit t. Replace
only this nonpersistent buffer value by-23.999878, leaving every weight fixed.
At Heun stages t=i/N for Ndividing1024, the two phases differ by an integer
multiple of2*pi. Their sine/cosine features and entire network functions agree
there in exact arithmetic. Checks at128/256/512/1024therefore share this blind spot.

All twelve checkpoints retain their source/config/data bindings. Each uses
1025grid predictions, all4096cached continuous-time field pairs, sixteen native
Heun128source endpoints and the original128-pair/five-time selection Jacobians.
Both FP32andFP64are checked. The frequency and all tensors are restored exactly.

{table}

## Observations

FP64grid predictions agree within{maxima['grid_max_difference']:.2e}, and native
endpoints within{maxima['endpoint_max_difference']:.2e}. FP32endpoint differences
are at most{maxima['fp32_endpoint_max_difference']:.2e}. The off-grid velocity
change is{min(ratios):.3f}--{max(ratios):.3f}of each model's oracle-fitting RMSerror.
This measures learned use of the feature rather than inferring use from its
presence. Raw off-grid fitting MSEs remain in rows.json.

![Different responses under the same sampler](figures/same-sampler-different-response.png)

For Ring8Free ISRU, the measured selection Jacobian error changes0.306->0.344
while FP64native endpoints remain the same to numerical rounding. The focal
palindrome/alternating ordering does not reverse under this intervention.
Thus the earlier spatial association remains an observation, but cannot alone
identify a continuous-field cause of the sampler's output.

![Both actual precisions](figures/rounding-level-sample-differences.png)

The construction is exact over real arithmetic. Finite FP32/FP64differences
are reported separately, not called bit-identical. The endpoint witness uses
sixteen sources per checkpoint; the algebra, not a sixteen-sample performance
estimate, explains the shared-grid identity. No new SW2orFIDis claimed.

## Consequence and next action

Keep all original fixed-grid sampling scores. Restrict earlier outer-resolution
claims to the tested dyadic grids. The separate256-source sensitivity calibration
also remains numerically unresolved at its1024cap, so it cannot yet establish
whether the linear transport explanation works. The alias construction does not
prove that this one frequency causes those failures.

Next qualify native fields on non-dyadic and above-bandwidth grids before a
continuous-flow interpretation. Keep weights, hidden integration, source arrays
and ordinary controls fixed. Do not introduce an activation, optimizer or new
training recipe to repair a numerical explanation.

## Verification and claim scope

The fresh documented native and independent NumPy commands exit0/0. Independent
summary reconstruction agrees within{proof['max_error']:.2e}; raw hashes and all
twelve cases are retained. No across-model significance test is needed for this
deterministic construction. It does not establish training reliability, general
generation improvement or an ODE-specific advantage.
''')
    (out/'stats-appendix.md').write_text('''# Statistical scope

This is a deterministic function/sampler counterexample on twelve existing
checkpoints. It uses4096cached field pairs,1025grid states and16endpoint sources
per checkpoint; the separate Jacobian comparison retains128pairs/five times.
There are no new training runs or independent-training confidence intervals.
No p-values are used. Means/RMSvalues describe the fixed input population.
The exact phase identity is mathematical; finite precision and endpoint
witnesses have explicit tolerances and complete raw predictions.
''')
    (out/'figure-catalog.md').write_text('''# Figure catalogue

- same-sampler-different-response: every original/alias Jacobian pair on the
  same128-pair/five-time grid. No error bars or new model selection. Notice that
  the metric can change while native sampling is preserved. This limits causal
  interpretation, without reversing the observed focal ordering.
- rounding-level-sample-differences: maximum endpoint change across sixteen
  fixed sources per checkpoint, actual FP32andFP64results. The logarithmic axis
  separates rounding scales. This verifies the construction rather than ranking
  model performance or extrapolating a sixteen-sampleSW2estimate.
''')
    dump(out/'COMPLETE.json', dict(passed=True, cases=12, source_sha256=sha(Path(__file__)),
        artifacts={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}))
    print('Complete twelve-model temporal alias audit:', maxima, flush=True)


if __name__ == '__main__':
    main()
