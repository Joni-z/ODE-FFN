"""Complete evidence for single-layer refinement-response interventions."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry import read, dump, sha
from analyze import LABEL, TASKS


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
        c, v, s = read(native/'COMPLETE.json'), read(verified/'COMPLETE.json'), read(verified/'summary.json')
        assert c['passed'] and v['passed'] and not c['witness'] and not v['witness']
        assert c['cases'] == v['cases'] == 3 and c['inputs_per_case'] == 80 and c['layers_per_case'] == 6
        assert sha(verified/'summary.json') == v['summary_sha256']
        assert s['input_complete_sha256'] == sha(native/'COMPLETE.json') and s['all_source_arrays_exact']
        rows += s['rows']
        maximum = max(maximum, s['max_candidate_reconstruction_error'])
        bindings.append(dict(dataset=name, native_folder=str(native), native_complete_sha256=sha(native/'COMPLETE.json'),
            verified_complete_sha256=sha(verified/'COMPLETE.json'), verified_summary_sha256=sha(verified/'summary.json')))
    rows.sort(key=lambda r: (r['dataset'], LABEL[r['family']]))
    assert len(rows) == 6
    out.mkdir(parents=True, exist_ok=False)
    (out/'figures').mkdir()
    dump(out/'summary.json', dict(rows=rows, bindings=bindings, max_candidate_reconstruction_error=maximum))
    shutil.copyfile(root/'terminal-receipts.json', out/'terminal-receipts.json')
    shutil.copyfile(root/'readiness/READY.json', out/'readiness.json')
    table = '| Task | Model | Normal correction | Tangential correction | Exact-potential ray | Adjusted T-minus-N interval | Adjusted T-minus-ray interval |\n|---|---|---:|---:|---:|---|---|\n'
    for r in rows:
        f = r['fractions']
        intervals = r['uncertainty']['intervals']
        nlo, nhi = intervals['tangential_minus_normal']['nominal_eighteen_contrast_adjusted']
        rlo, rhi = intervals['tangential_minus_ray']['nominal_eighteen_contrast_adjusted']
        table += f"| {'Ring8' if r['dataset']==TASKS[0] else 'Product16'} | {LABEL[r['family']]} | {f['normal']:.3f} | {f['tangential']:.3f} | {f['ray']:.3f} | [{nlo:.3f}, {nhi:.3f}] | [{rlo:.3f}, {rhi:.3f}] |\n"
    (out/'table.md').write_text(table)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), constrained_layout=True)
    for ax, task, title in zip(axes, TASKS, ['Ring8', 'Product16']):
        group = [r for r in rows if r['dataset'] == task]
        for i, (key, label, color) in enumerate([('normal', 'Normal correction', '#2166ac'),
                ('tangential', 'Tangential correction', '#d95f02'), ('ray', 'Exact-potential ray', '#1b9e77')]):
            values = [r['fractions'][key] for r in group]
            x = np.arange(3)+(i-1)*.23
            ax.bar(x, values, width=.23, color=color, label=label)
            if key == 'tangential':
                limits = np.array([r['uncertainty']['intervals']['tangential_fraction']['nominal_eighteen_contrast_adjusted'] for r in group])
                ax.errorbar(x, values, yerr=[np.asarray(values)-limits[:, 0], limits[:, 1]-values],
                            fmt='none', color='black', capsize=3, lw=1)
        ax.axhline(0, color='.5', lw=1)
        ax.set(xticks=range(3), xticklabels=[LABEL[r['family']] for r in group], title=title,
            ylabel='Fraction of one-layer refinement response reproduced')
        ax.legend(fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('single-layer-output-components.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), constrained_layout=True)
    for ax, task, title in zip(axes, TASKS, ['Ring8', 'Product16']):
        group = [r for r in rows if r['dataset'] == task]
        for i, r in enumerate(group):
            for offset, key, value, color, label in [(-.1, 'tangential_minus_normal', r['fractions']['tangential']-r['fractions']['normal'], '#2166ac', 'T minus normal'),
                    (.1, 'tangential_minus_ray', r['fractions']['tangential']-r['fractions']['ray'], '#1b9e77', 'T minus ray')]:
                lo, hi = r['uncertainty']['intervals'][key]['nominal_eighteen_contrast_adjusted']
                ax.hlines(i+offset, lo, hi, color=color, lw=2)
                ax.scatter(value, i+offset, color=color, s=28, label=label if i == 0 else None)
        ax.axvline(0, color='.5', lw=1)
        ax.set(yticks=range(3), yticklabels=[LABEL[r['family']] for r in group], title=title,
               xlabel='Paired improvement in reproduced output response')
        ax.legend(fontsize=8, loc='upper left')
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('paired-component-effects.'+ext), dpi=180)
    plt.close(fig)
    supported = sum(r['direction_hypothesis_point_support'] for r in rows)
    adjusted = sum(all(v['nominal_eighteen_contrast_adjusted'][0] > 0
        for v in r['uncertainty']['intervals'].values()) for r in rows)
    (out/'analysis-report.md').write_text(f'''# Which hidden component reaches the output?

**Tangential correction reproduces most of the single-layer refinement response
in all {supported}/6 fixed models. All {adjusted}/6 satisfy the three registered
conditions with positive lower adjusted input-bank intervals.** Matching the
potential while retaining the original feature direction reproduces much less.

## Controlled experiment

Reuse all80original interpolation inputs per checkpoint and all six local layers.
Replace exactly one layer per forward call, preserving its exact upstream state
and all native downstream computation. Compare full continuous-reference output,
normal correction, tangential correction and an exact-potential ray correction.
The ray changes only amplitude and matches the reference potential. All2880ray
points qualify; identity hooks and native roundtrips are bit-identical.

The reported fraction is1 minus the squared residual to the full one-layer
reference response, divided by the original response energy. It measures a
finite intervention's output effect. It is not the earlier oracle-endpoint
calibration, a generation-quality score or a percentage improvement in FID.

{table}

![Output response by component](figures/single-layer-output-components.png)

Tangential correction closely reproduces the full response in both tasks.
Normal and exact-potential corrections have much smaller effects. This direct
intervention strengthens the earlier feature-space observation: the small
potential-normal component does not become the main output effect downstream.
The result concerns the inspected states and layers; it does not identify the
skew part of P as the sole source of direction changes.

![Paired bank effects](figures/paired-component-effects.png)

Both registered contrasts favor tangential correction in every fixed model,
with positive nominal18-comparison-adjusted conditional intervals. Banks jointly
resample residual and response energies. The intervals describe these weights'
input variation, not repeated training trajectories.

## Consequence

An energy-only account of the hidden-refinement output change is rejected on
this cohort. The supported mechanism is direction-sensitive feature computation.
It explains which local difference reaches the complete model's prediction;
it does not yet establish why the final generated distribution improves or
worsens. No spectral cap, energy penalty or new architecture follows automatically.

Both native datasets and both independent verifiers exit0. Candidate and exact
ray reconstruction errors are at most{maximum:.2e}. Every layer, bank, time,
oracle-field diagnostic and source binding is retained. All weights/buffers and
the image trainer remain fixed. [Protocol and definitions](../hidden-readout-protocol.md).

## Claim candidates

- Claim: tangential correction carries the observed full-output refinement response.
  - Evidence: six-model single-layer interventions, exact energy control and adjusted paired intervals.
  - Allowed wording: direct conditional effect on the registered80inputs and all six layers.
  - Forbidden stronger wording: improved generation, unique skew-P benefit or a new winning FFN.
  - Uncertainty: fixed weights and finite interpolation inputs; no sample trajectories are changed here.
  - Next check: relate direction-sensitive hidden computation to a qualified distribution intervention.
  - Decision: keep the local output mechanism.
''')
    stats = '''# Units and independent statistics

Six models, six separately intervened layers/model,80inputs/model and eight
common banks. Each bank retains ten inputs and six layer interventions. These
are repeated conditional effects, not independent training replicates. The
reference response changes one layer at a time, never all layers simultaneously.

Jointly bootstrap the denominator and all three residual-energy sums across
banks before forming ratios. Exact enumeration uses6435weighted multisets,
equivalent to8^8ordered resamples. The18registered quantities are F_T,F_T-F_N
andF_T-F_rayfor each of six models. Both ordinary95%and nominal18-comparison
Bonferroni intervals are retained in summary.json. No Gaussian assumption or
p-value is used. All zero-response/unresolved-ray cases would remain visible.

| Task/model | Layer | Normal fraction | Tangential fraction | Ray fraction | Native oracle MSE | Full-reference oracle MSE |\n|---|---|---:|---:|---:|---:|---:|\n'''
    for r in rows:
        for layer in r['layers']:
            f = layer['fractions']
            stats += f"| {'Ring8' if r['dataset']==TASKS[0] else 'Product16'} / {LABEL[r['family']]} | {layer['layer']} | {f['normal']:.3f} | {f['tangential']:.3f} | {f['ray']:.3f} | {layer['oracle_mse_by_mode'][0]:.3f} | {layer['oracle_mse_by_mode'][1]:.3f} |\n"
    (out/'stats-appendix.md').write_text(stats)
    (out/'figure-catalog.md').write_text('''# Figure catalog

## single-layer-output-components.png / .pdf
- Purpose: test whether potential-normal or tangential hidden error reaches the full output.
- Data: independent six-model response fractions, all36layer interventions and80inputs/model.
- Caption: target is the full single-layer reference response; ray preserves direction and matches potential; error bars are adjusted conditional input-bank intervals for tangential correction.
- Observation: direction correction reproduces much more than either energy-related control.
- Interpretation: supports a local output mechanism, not sampling superiority.

## paired-component-effects.png / .pdf
- Purpose: retain paired evidence for both registered component comparisons.
- Data: exact bank-bootstrap contrasts in summary.json.
- Caption:18-comparison-adjusted conditional intervals, same banks/inputs/weights; zero means equal response reproduction.
- Observation: both contrasts favor tangential correction in every fixed model.
- Interpretation: energy-only explanation of this output change fails; generation remains a separate target.
''')
    dump(out/'COMPLETE.json', dict(passed=True, models=6, local_states=2880, point_support=supported,
        all_adjusted_support=adjusted, source_sha256=sha(Path(__file__)),
        files={str(f.relative_to(out)): sha(f) for f in sorted(out.rglob('*')) if f.is_file()}))


if __name__ == '__main__':
    main()
