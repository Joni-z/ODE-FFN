"""Complete equal-dimension control: separate geometry from special response."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    rows, bindings, max_error = [], [], 0.
    for task in ['ring', 'product']:
        native, verified = root/'formal'/task, root/'formal'/(task+'-verified')
        complete, proof, summary = read(native/'COMPLETE.json'), read(verified/'COMPLETE.json'), read(verified/'summary.json')
        assert complete['passed'] and proof['passed'] and not complete['witness'] and not proof['witness']
        assert complete['cases'] == proof['cases'] == 3 and complete['layers_per_case'] == 6
        assert complete['inputs_per_case'] == 80 and complete['directions'] == 64
        assert sha(verified/'summary.json') == proof['summary_sha256']
        assert summary['input_complete_sha256'] == sha(native/'COMPLETE.json') and summary['frames_reconstruct']
        rows += summary['rows']
        max_error = max(max_error, summary['max_candidate_reconstruction_error'])
        bindings.append(dict(task=task, native_complete_sha256=sha(native/'COMPLETE.json'),
            verified_summary_sha256=sha(verified/'summary.json'), seconds=complete['seconds']))
    assert len(rows) == 6
    tangent_negative = sum(row['contrasts'][1] < 0 for row in rows)
    tangent_adjusted_negative = sum(row['uncertainty']['intervals']['tangent_minus_null']['nominal_twelve_contrast_adjusted'][1] < 0 for row in rows)
    normal_adjusted_positive = sum(row['uncertainty']['intervals']['normal_minus_null']['nominal_twelve_contrast_adjusted'][0] > 0 for row in rows)
    out.mkdir(parents=True, exist_ok=False)
    (out/'figures').mkdir()
    dump(out/'summary.json', dict(rows=rows, bindings=bindings, tangent_negative_cases=tangent_negative,
        tangent_adjusted_negative_cases=tangent_adjusted_negative,
        normal_adjusted_positive_cases=normal_adjusted_positive,
        max_candidate_reconstruction_error=max_error))
    shutil.copyfile(root/'terminal-receipts.json', out/'terminal-receipts.json')
    shutil.copyfile(root/'readiness/READY.json', out/'readiness.json')
    shutil.copyfile(root/'frames/manifest.json', out/'frames.json')
    table = '| Task | Model | Potential normal F | Random 1D mean F | Potential tangent F | Random 63D mean F | Delta N | Delta T |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    interval_table = '| Task | Model | Delta N adjusted interval | Delta T adjusted interval |\n|---|---|---|---|\n'
    gains = '| Task | Model | Actual normal gain | Null 1D gain | Normal gain ratio | Actual tangent gain | Null 63D gain |\n|---|---|---:|---:|---:|---:|---:|\n'
    for row in rows:
        task, model = ('Ring8' if row['dataset'] == TASKS[0] else 'Product16'), LABEL[row['family']]
        a, n, d = row['actual_fractions'], row['null_mean_fractions'], row['contrasts']
        table += f'| {task} | {model} | {a[0]:.3f} | {n[0]:.3f} | {a[1]:.3f} | {n[1]:.3f} | {d[0]:.3f} | {d[1]:.3f} |\n'
        lo_n, hi_n = row['uncertainty']['intervals']['normal_minus_null']['nominal_twelve_contrast_adjusted']
        lo_t, hi_t = row['uncertainty']['intervals']['tangent_minus_null']['nominal_twelve_contrast_adjusted']
        interval_table += f'| {task} | {model} | [{lo_n:.3f}, {hi_n:.3f}] | [{lo_t:.3f}, {hi_t:.3f}] |\n'
        ag, ng = row['actual_finite_response_gain'], row['null_finite_response_gain']
        gains += f'| {task} | {model} | {ag[0]:.3f} | {ng[0]:.3f} | {ag[0]/ng[0]:.3f} | {ag[1]:.3f} | {ng[1]:.3f} |\n'
    (out/'table.md').write_text(table)
    (out/'intervals.md').write_text(interval_table)
    (out/'finite-response-gains.md').write_text(gains)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.3))
    for column, (task, title) in enumerate(zip(TASKS, ['Ring8', 'Product16'])):
        group = [row for row in rows if row['dataset'] == task]
        for component in range(2):
            ax = axes[component, column]
            for index, row in enumerate(group):
                actual, null = row['actual_fractions'][component], row['null_mean_fractions'][component]
                ax.plot([index-.10, index+.10], [actual, null], color='.65', lw=1)
                ax.scatter(index-.10, actual, color='#2166ac', marker='o', s=42, label='Potential split' if index == 0 else None)
                ax.scatter(index+.10, null, color='#d95f02', marker='s', s=38, label='Complete-frame mean' if index == 0 else None)
            ax.set(xticks=range(3), xticklabels=[LABEL[row['family']] for row in group],
                ylabel='Output refinement reproduced (F)', title=title+(' / 1D normal' if component == 0 else ' / 63D complement'))
            ax.set_ylim((0, .28) if component == 0 else (.89, 1.005))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .005), ncol=2)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('dimension-matched-response.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9))
    for ax, task, title in zip(axes, TASKS, ['Ring8', 'Product16']):
        group = [row for row in rows if row['dataset'] == task]
        for index, row in enumerate(group):
            for component, key, offset, color, label in [(0, 'normal_minus_null', -.1, '#2166ac', 'Potential normal minus 1D null'),
                    (1, 'tangent_minus_null', .1, '#d95f02', 'Potential tangent minus 63D null')]:
                lo, hi = row['uncertainty']['intervals'][key]['nominal_twelve_contrast_adjusted']
                ax.hlines(index+offset, lo, hi, color=color, lw=2)
                ax.scatter(row['contrasts'][component], index+offset, color=color, s=30, label=label if index == 0 else None)
        ax.axvline(0, color='.4', lw=1)
        ax.set(yticks=range(3), yticklabels=[LABEL[row['family']] for row in group], title=title,
               xlabel='Paired change in reproduced response')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .005), ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, 1))
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('paired-dimension-contrasts.'+ext), dpi=180)
    plt.close(fig)
    (out/'analysis-report.md').write_text(f'''# Equal-dimension controls change the mechanism interpretation

**The strong tangential-specificity hypothesis fails in {tangent_negative}/6
models. Every adjusted paired interval is negative in {tangent_adjusted_negative}/6.**
A generic 63D correction reproduces more of the full local-refinement output
change than the potential-tangential correction. Large tangential F therefore
does not by itself establish unusual ODE sensitivity.

## Purpose and fixed intervention

Keep the six trained checkpoints, all six local layers and all 80 original
inputs. Before observing these outputs, bind one complete orthonormal frame
per model/layer using evaluation RNG 0. For every one of its 64 directions,
remove the 1D projection of the same native-minus-reference feature difference,
or remove its 63D complement. Compare with the prior exact potential-normal
and potential-tangential interventions, with the same downstream network.
No weights, response function, data or image architecture are changed.

![Equal-dimension response](figures/dimension-matched-response.png)

The top row compares two 1D corrections; the bottom compares two 63D corrections.
Points, rather than bar lengths, show the high-F complement results. Complete
frames have exactly the expected mean feature-removal energy at every state.
The potential normal captures more of the output change, while its complement
captures less than the corresponding dimension-matched mean.

{table}

## Paired conditional evidence

![Paired dimension contrasts](figures/paired-dimension-contrasts.png)

Intervals enumerate all eight-bank resamples and adjust nominally for the 12
registered contrasts. Tangent-minus-null is negative in every case. Normal
specificity has positive adjusted intervals in {normal_adjusted_positive}/6
models, all on Ring8; Product16 intervals cross zero.

{interval_table}

The earlier observation that tangential changes carry most output response
remains true, but its interpretation changes: a generic high-dimensional
complement carries even more. The potential-normal coordinate is structured,
especially on Ring8, despite being only one of 64 dimensions.

## Feature error concentration and finite response gain

The preceding exact geometry audit found potential-normal feature error
enriched by 2.158--16.489 relative to the 1/64 geometric reference. That alone
can increase its absolute output effect. Dividing squared output displacement
by squared removed feature length also reveals a task difference: Ring8's
normal gains exceed the null, whereas Product16's are lower. These are finite
intervention gains, not Jacobian estimates or a linear scaling law.

{gains}

## Verification and research decision

Both native datasets and both independent verifiers exit 0. All 2880 original
states, all 36 frames and all 64 directions are retained. Frame generation,
candidate projections, exact mean feature-energy identities, full-output
metrics and bank intervals reconstruct independently; maximum candidate
reconstruction error is {max_error:.3e}. Identity upstream states, original
normal/tangent outputs and final native roundtrips are exact.

Do not promote a direction-only architecture or a larger generation comparison
from high tangential F/G. The next mathematical question is why the trained
hidden solver's error concentrates along the potential normal, and why the
remaining network treats that direction differently across tasks. A quantitative
solver-bias prediction is more informative than another activation search.

## Claim candidates

- Claim: the tested potential tangent is not superior to an equal-dimension
  generic complement on these fixed models and inputs.
  - Evidence: all six paired contrasts and their adjusted negative intervals.
  - Allowed wording: the strong six-case tangential-specificity prediction is
    falsified on this complete fixed cohort.
  - Forbidden stronger wording: no ODE feature computation can ever be useful.
  - Uncertainty: one bound complete frame per layer, fixed checkpoints and inputs.
  - Next check: predict the actual hidden discretization error quantitatively.
  - Decision: discard tangential dominance alone as a novelty claim.
- Claim: the potential normal carries above-null output response on Ring8.
  - Evidence: three positive adjusted paired intervals, feature enrichment and
    the supporting finite-response gains; Product16 remains inconclusive.
  - Allowed wording: task-dependent canonical-normal structure.
  - Forbidden stronger wording: general generation or image-quality improvement.
  - Uncertainty: no training-population inference or exclusive ODE advantage.
  - Next check: separate numerical bias from downstream feature sensitivity.
  - Decision: retain the mathematical mechanism question.
''')
    (out/'stats-appendix.md').write_text(f'''# Statistical units and complete comparisons

Six fixed checkpoints, 80 inputs each, six separate layers and all 64 frame
directions. Evaluation RNG 0 prepares 36 data-independent orthonormal frames
before new outputs. Directions are correlated members of a complete frame;
they are not independent statistical samples. The eight original input banks
are the conditional uncertainty units, retaining ten inputs per bank and all
layers/directions. No training-population inference is made.

F uses the same full-reference response denominator. Null residual energies
are averaged over every direction before pooling, without a selected direction.
Delta N/T are actual minus null F. Exact bank bootstrap enumerates 6435 weighted
multisets representing all 8^8 ordered draws. Joint numerator/denominator
resampling preserves pairing. Ordinary 95% and nominal 12-contrast-adjusted
intervals are retained in summary.json. Adjusted two-sided tail probability
is 0.05/24. Zero denominators remain unresolved rather than excluded.

The strong tangential hypothesis requires positive Delta T in every model;
all {tangent_negative} point contrasts and {tangent_adjusted_negative} adjusted
intervals instead have negative sign. Normal specificity is a separate
hypothesis, with {normal_adjusted_positive} resolved positive adjusted intervals.
The frame means and paired F differences are the effect sizes. The finite
response gains divide squared field displacement by squared removed feature
length and do not assume local linearity. Raw precision is retained.
''')
    (out/'figure-catalog.md').write_text('''# Figure catalog

## dimension-matched-response
Purpose: test whether potential components outperform generic components of
the same dimensions. Data: exact pooled actual and 64-direction mean F for all
six cases. Top: 1D normal; bottom: 63D complement. Point markers use separate
vertical scales to resolve the complement values; no bar-length comparison.
No frame-replicate error bars are implied. Inspect the complete paired-interval
figure before making a resolved comparison. High tangent F alone is insufficient.

## paired-dimension-contrasts
Purpose: assess the two separately registered specificity hypotheses. Data:
actual-minus-null F with nominal 12-contrast-adjusted conditional input-bank
intervals. Blue is normal; orange tangent. Zero is equality. Notice all negative
tangent intervals, Ring8 positive normal intervals and Product16 intervals
crossing zero. This changes the next mechanism test; it is not a quality ranking.
''')
    print('Complete null:', tangent_adjusted_negative, 'negative adjusted tangent contrasts;',
          normal_adjusted_positive, 'positive adjusted normal contrasts', flush=True)


if __name__ == '__main__':
    main()
