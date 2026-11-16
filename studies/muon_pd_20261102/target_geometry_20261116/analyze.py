"""Figures and analysis for the registered frozen target-Jacobian diagnostic."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry import read, sha, dump

LABEL = {'state_midpoint4': 'Shared P', 'stage_operator_midpoint4': 'Released stages',
    'palindrome_operator_midpoint4': 'Palindrome', 'alternating_operator_midpoint4': 'Alternating',
    'free_isru': 'Free ISRU', 'swiglu64': 'SwiGLU64/Muon', 'swiglu192': 'SwiGLU192/AdamW'}
COLORS = {'Shared P': '#2166ac', 'Released stages': '#762a83', 'Palindrome': '#1b9e77',
          'Alternating': '#d95f02', 'Free ISRU': '#7570b3', 'SwiGLU64/Muon': '#666666',
          'SwiGLU192/AdamW': '#666666'}
TASKS = ['gmm_ring8_2d', 'rotated_product_bimodal_16d']


def fmt(value: float) -> str:
    return f'{value:.3e}' if 0 < abs(value) < .0005 else f'{value:.3f}'


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ['input', 'verified', 'output']:
        p.add_argument('--'+key, required=True)
    args = p.parse_args()
    raw, verified, out = map(Path, [args.input, args.verified, args.output])
    proof, result = read(verified/'COMPLETE.json'), read(verified/'summary.json')
    assert proof['passed'] and not proof['witness'] and proof['rows'] == 60
    assert sha(verified/'summary.json') == proof['summary_sha256']
    native = read(raw/'COMPLETE.json')
    assert native['passed'] and not native['witness'] and native['points_per_time'] == 128
    out.mkdir(parents=True, exist_ok=False)
    (out/'figures').mkdir()
    for source, name in [(verified/'summary.json', 'summary.json'),
        (verified/'COMPLETE.json', 'independent-COMPLETE.json'), (raw/'COMPLETE.json', 'native-COMPLETE.json'),
        (raw/'rows.json', 'native-rows.json')]:
        shutil.copyfile(source, out/name)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for axis, dataset, title in zip(axes, TASKS, ['Ring8', 'Product16']):
        group = [r for r in result['rows'] if r['dataset'] == dataset]
        for row in group:
            label = LABEL[row['family']]
            axis.scatter(row['velocity_jacobian_error'], row['confirmation_pooled_sw2'],
                         color=COLORS[label], label=label, s=45)
        axis.set(xlabel='Full velocity-Jacobian error (lower is better)',
                 ylabel='Confirmation pooled SW2 (lower is better)', title=title)
        axis.legend(fontsize=8, loc='best')
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('jacobian-versus-sampling.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for axis, dataset, title in zip(axes, TASKS, ['Ring8', 'Product16']):
        for row in [r for r in result['rows'] if r['dataset'] == dataset]:
            label = LABEL[row['family']]
            axis.plot([v['time'] for v in row['by_time']],
                      [v['velocity_jacobian_error'] for v in row['by_time']],
                      'o-', label=label, color=COLORS[label], markersize=4)
        axis.set(xlabel='Generative interpolation time t', ylabel='Full velocity-Jacobian error', title=title)
        axis.legend(fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('time-resolved-jacobian.'+ext), dpi=180)
    plt.close(fig)
    table = '| Task | Model | Field MSE | Full J error | Image skew energy | Negative symmetric energy | Pooled SW2 |\n|---|---|---:|---:|---:|---:|---:|\n'
    for row in result['rows']:
        values = [row[k] for k in ['field_mse', 'velocity_jacobian_error', 'image_skew_energy',
                                   'image_symmetric_negative_energy', 'confirmation_pooled_sw2']]
        table += f"| {'Ring8' if row['dataset']==TASKS[0] else 'Product16'} | {LABEL[row['family']]} | "+' | '.join(map(fmt, values))+' |\n'
    pairs = '| Task | Palindrome J error | Alternating J error | Difference | Expected sign | SW2 difference |\n|---|---:|---:|---:|---|---:|\n'
    intervals = []
    for row in result['matched_pairs']:
        task = 'Ring8' if row['dataset'] == TASKS[0] else 'Product16'
        pairs += f"| {task} | {fmt(row['palindrome'])} | {fmt(row['alternating'])} | {fmt(row['primary_gap'])} | {'negative' if row['expected_gap_sign']<0 else 'positive'} | {fmt(row['sampling_gap'])} |\n"
        intervals.append(f"- {task}: paired gap {fmt(row['primary_gap'])}, SD {fmt(row['point_pair_difference_sd'])}; conditional95% interval {list(map(fmt,row['conditional_pair_bootstrap95']))}; two-contrast interval {list(map(fmt,row['conditional_two_contrast_bonferroni']))}.")
    decision = ('The registered point direction holds on both tasks, but both conditional input-bootstrap intervals span zero. The association remains unresolved at this selection budget.'
                if result['primary_explanation_passes_both_tasks'] else
                'Reject the registered explanation that lower average full velocity-Jacobian error explains both matched schedule effects. Retain each measured task direction; do not replace the primary metric with a favorable time or secondary spectrum summary.')
    (out/'table.md').write_text(table)
    (out/'analysis-report.md').write_text(f'''# Frozen target-geometry observation

**{decision}**

## Question and unit

The prior matched schedules improve sampling in opposite task directions.
This observation asks whether full output response error reproduces that pattern.
All twelve existing20k checkpoints are retained. Each uses128 common selection
source/target pairs at five fixed times:640 interpolation states. Training is
seed0. The twelve checkpoints are six model/settings per task, not twelve
independent training replicates. The ordinary controls retain their documented
optimizer settings. These synthetic velocity MLPs are not the image JiT model.

## Registered primary comparison

Full J error is mean ||J_v-J_v*||_F²/d, equally weighted across points and times.
Difference is palindrome minus alternating. Sampling scores come from the
existing common-bank confirmation cohort; no samples were regenerated here.

{pairs}

## Complete context

{table}

![Complete checkpoint comparison](figures/jacobian-versus-sampling.png)

Each point is an existing trained checkpoint, with no fitted regression line.
Read the matched schedule pair first. Cross-family locations are descriptive
and do not isolate architecture from optimizer or capacity. Image skew and
negative symmetric energy concern J_m=I+(1-t)J_v; eigenvalues are of sym(J_m),
not singular values or eigenvalues of an arbitrary nonsymmetric Jacobian.
On Product16, SwiGLU has the best sampling here despite the highest average
Jacobian error. Therefore this metric does not provide a general cross-model
ranking, even though the two focal point directions agree.

![All five registered times](figures/time-resolved-jacobian.png)

The time profile retains every registered slice. It can locate response errors
but cannot override the equal-time primary decision. These are interpolation
states, not generated trajectories, so this observation cannot establish which
errors actually accumulate along sampling paths.

## Verification and resulting action

Native predictions, restored runtime buffers, FP64 conversion, central-difference
JVPs, batch separability and unchanged tensors all pass for twelve models.
Separate NumPy reconstruction agrees within {result['reconstruction_max_error']:.2e}.
See native-rows.json for all bindings and actual precision errors. Native oracles
and mathematical counterexamples are checked separately before this observation.

The next design decision must address the full task, not an internal spectrum
proxy. A failed primary association redirects measurement toward trajectory-
weighted error and mean bias; a passed association first needs a matched causal
intervention. Neither outcome alone promotes another architecture to CelebA.

## Claim candidates

- Claim: the exact Bayes target supplies a measurable whole-output response.
  Source: derivation.md and the native oracle checks. Decision: keep.
- Claim: lower average output-Jacobian error explains the schedule sampling gap.
  Source: the registered paired table above. Decision: {'unresolved association; confirm on independent inputs' if result['primary_explanation_passes_both_tasks'] else 'reject at this fixed-checkpoint protocol'}.
- Claim: hidden ODE structure provides a general image-generation benefit.
  Source: absent in this observation. Decision: unsupported; no image promotion.
''')
    (out/'stats-appendix.md').write_text('''# Statistical unit and exact uncertainty

The primary observation is a deterministic summary on128 paired interpolation
paths at fixed weights. Each pair is reused at five times. Average times within
each pair before resampling; do not treat640 rows as independent. The two-task
decision uses the registered point-gap direction, not a post-hoc p-value gate.

Conditional paired bootstrap:10000 resamples of128 source/target pair indices,
analysis RNG0, percentile intervals. The97.5% intervals use a two-contrast
Bonferroni family. SD describes point-pair differences. These intervals concern
input sampling at fixed checkpoints, not independent-training uncertainty.
No parametric normality assumption, t-test, across-model significance or
training-repeat confidence interval is used. No p-values are claimed.

'''+ '\n'.join(intervals)+'''

All twelve models and sixty time rows are present. Existing pooledSW2 values
retain their original32768-sample common-bank identity. Derivative states use
the selection cohort; SW2 is the separate preexisting confirmation cohort.
Different task scores and historical cohorts must not be pooled.
''')
    (out/'figure-catalog.md').write_text('''# Figure catalogue

## jacobian-versus-sampling.png / .pdf

Purpose: compare the prespecified whole-output response metric with generation
on all twelve checkpoints. Data: independently reconstructed summary.json,
confirmation scores bound in native-rows.json. One point per checkpoint; no
error bars or independent-training inference. Notice whether the focal pair
orders the same way on both axes. This determines whether the proposed average
response-error explanation survives; it does not assign causality.

## time-resolved-jacobian.png / .pdf

Purpose: expose every registered time slice behind the equal-time summary.
Data: summary.json/by_time. Each point averages128 common interpolated states;
the five times reuse the same pairs. No error bars or independent-time inference.
Notice where errors differ, while retaining the primary average decision.
This distinguishes an overall failed explanation from a possible new,
explicitly registered trajectory/time-dependent question.
''')
    dump(out/'COMPLETE.json', dict(passed=True, cases=12, time_rows=60,
        primary_explanation_passes_both_tasks=result['primary_explanation_passes_both_tasks'],
        analysis_source_sha256=sha(Path(__file__)),
        artifacts={str(f.relative_to(out)): sha(f) for f in out.rglob('*') if f.is_file()}))
    print(decision, flush=True)


if __name__ == '__main__':
    main()
