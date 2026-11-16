"""Complete finite-source precision evidence, figures and input-bank summaries."""
import argparse
import itertools
import math
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry import read, dump, sha
from analyze import LABEL, COLORS, TASKS


def bank_interval(rms: list[float]) -> dict:
    means = np.square(rms)
    values, weights = [], []
    for sequence in itertools.combinations_with_replacement(range(8), 8):
        counts = np.bincount(sequence, minlength=8)
        values.append(float(np.sqrt(counts@means/8)))
        weights.append(math.factorial(8)//math.prod(math.factorial(int(c)) for c in counts))
    order = np.argsort(values)
    cdf = np.cumsum(np.asarray(weights)[order])
    at = lambda q: float(np.asarray(values)[order][np.searchsorted(cdf, math.ceil(q*8**8))])
    return dict(bank_mean=float(np.mean(rms)), bank_std=float(np.std(rms, ddof=1)),
        conditional95=[at(.025), at(.975)], ordered_bootstrap_resamples=8**8)


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ['input', 'output']:
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    root, out = Path(a.input), Path(a.output)
    proof, source = read(root/'verified/COMPLETE.json'), read(root/'verified/summary.json')
    queue = read(root/'formal/queue-finished.json')
    assert proof['passed'] and proof['cases'] == 12 and len(source['rows']) == 12
    assert sha(root/'verified/summary.json') == proof['summary_sha256']
    assert sha(root/'formal/queue-finished.json') == proof['queue_terminal_sha256']
    assert queue['passed'] and len(queue['completed']) == 12 and not queue['unstarted']
    assert all(r['exit_code'] == 0 for r in queue['completed'])
    assert all(r['actual_exit_code'] == 0 for r in proof['receipts'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'figures').mkdir()
    rows = []
    for verified in sorted(source['rows'], key=lambda r: (r['dataset'], LABEL[r['family']])):
        binding = next(b for b in source['bindings'] if b['id'] == verified['id'])
        folder = Path(binding['native_folder'])
        assert sha(folder/'COMPLETE.json') == binding['native_complete_sha256']
        assert sha(folder/'rows.json') == binding['native_rows_sha256']
        native = read(folder/'rows.json')
        arrays = {}
        for level in native['levels']:
            file = folder/level['file']
            assert sha(file) == level['sha256']
            with np.load(file) as archive:
                arrays[level['steps']] = archive['endpoint'].copy()
        finest = arrays[verified['final_steps']]
        curve = [dict(steps=n, rms=float(np.sqrt(np.mean((endpoint-finest)**2))),
                      maximum=float(np.max(np.abs(endpoint-finest))))
                 for n, endpoint in sorted(arrays.items()) if n != verified['final_steps']]
        contrasts = []
        for contrast in verified['contrasts']:
            compact = {k: v for k, v in contrast.items() if not k.startswith('per_source_')}
            contrasts.append(dict(compact, **bank_interval(compact['bank_rms'])))
        rows.append(dict(id=verified['id'], dataset=verified['dataset'], family=verified['family'],
            qualified=verified['qualified'], final_steps=verified['final_steps'],
            final_gate=verified['final_gate'], contrasts=contrasts, convergence_to_finest=curve,
            seconds=read(folder/'COMPLETE.json')['seconds']))
    dump(out/'summary.json', dict(rows=rows, qualified_cases=source['qualified_cases'],
        independent_maximum_error=source['maximum_reconstruction_error'], bindings=source['bindings'],
        verified_summary_sha256=sha(root/'verified/summary.json')))
    for name, file in [('independent-COMPLETE.json', root/'verified/COMPLETE.json'),
                        ('queue-finished.json', root/'formal/queue-finished.json'),
                        ('queue-start.json', root/'formal/queue-start.json'),
                        ('readiness.json', root/'readiness/READY.json')]:
        shutil.copyfile(file, out/name)
    table = '| Task | Model | 1000 vs1024 RMS | 128 vsfinest RMS | 1024 vsfinest RMS | Finest | Final refinement RMS | Pass |\n|---|---|---:|---:|---:|---:|---:|---|\n'
    for r in rows:
        by = {(c['left'], c['right']): c for c in r['contrasts']}
        table += f"| {'Ring8' if r['dataset']==TASKS[0] else 'Product16'} | {LABEL[r['family']]} | {by[1000,1024]['rms']:.3e} | {by[128,r['final_steps']]['rms']:.3e} | {by[1024,r['final_steps']]['rms']:.3e} | {r['final_steps']} | {r['final_gate']['rms']:.3e} | {'yes' if r['qualified'] else 'no'} |\n"
    (out/'table.md').write_text(table)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)
    ticks = [128, 1000, 1024, 1032, 2048, 4096]
    for ax, dataset, title in zip(axes, TASKS, ['Ring8', 'Product16']):
        for r in [r for r in rows if r['dataset'] == dataset]:
            label = LABEL[r['family']]
            x = [ticks.index(c['steps']) for c in r['convergence_to_finest']]
            y = [c['rms'] for c in r['convergence_to_finest']]
            ax.plot(x, y, '.-', label=label, color=COLORS[label], lw=1.4)
        ax.set(title=title, xticks=range(len(ticks)), xticklabels=ticks, yscale='log',
            xlabel='Heun steps (categorical spacing)', ylabel='Endpoint RMS relative to own finest level')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('native-grid-convergence.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5), constrained_layout=True)
    pairs = [(1000, 1024), (1032, 1024), (1000, 1032)]
    for ax, dataset, title in zip(axes, TASKS, ['Ring8', 'Product16']):
        group = [r for r in rows if r['dataset'] == dataset]
        for j, (pair, color) in enumerate(zip(pairs, ['#2166ac', '#d95f02', '#1b9e77'])):
            for i, r in enumerate(group):
                c = next(c for c in r['contrasts'] if (c['left'], c['right']) == pair)
                x = i+(j-1)*.22
                ax.scatter(np.full(8, x), c['bank_rms'], color=color, s=10, alpha=.35)
                ax.scatter(x, c['rms'], color=color, marker='D', s=27,
                           label=f'{pair[0]} vs {pair[1]}' if i == 0 else None)
        ax.set(title=title, xticks=range(6), xticklabels=[LABEL[r['family']] for r in group],
               yscale='log', ylabel='Same-source endpoint RMS difference')
        ax.tick_params(axis='x', labelrotation=32)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('near-equal-work-grid-sensitivity.'+ext), dpi=180)
    plt.close(fig)
    count4096 = sum(r['qualified'] and r['final_steps'] == 4096 for r in rows)
    (out/'analysis-report.md').write_text(f'''# Native-time precision on all twelve frozen models

**All twelve native endpoint calculations meet the registered component gate:
{count4096}at4096steps and{12-count4096}at8192. Near-equal-work grids still give
different endpoints.** These are precision results at fixed weights, not a
new sampling-performance comparison.

## Purpose and population

Separate fixed-grid numerical behavior from the continuous-field interpretation
needed for the error-transport test. Each20kcheckpoint retains all256sources:
32from each of eight original selection banks. No source/model is dropped.
Internal integration, time encoding, weights and buffers remain unchanged.
The native Heun sampler computes1000/1032/2048/4096and8192only when needed;
bound128/1024arrays are reused exactly. Both ordinary controls remain included.

{table}

## Observations

![Native endpoint convergence](figures/native-grid-convergence.png)

Every row passes the original all-source componentwise atol1e-4/rtol1e-3gate
on its final refinement. Ring8released-stage and Product16SwiGLU need8192;
the others stop at4096. The plotted reference is each model's own finest
computed endpoint. Its zero by construction is omitted. A finite refinement
test establishes the registered numerical precision, not an exact solution.

![Nearby step counts](figures/near-equal-work-grid-sensitivity.png)

The1000/1024/1032contrasts measure grid sensitivity at nearly equal work;
they do not rank which grid is more accurate. Small points are eight bank RMSs;
diamonds are RMSover every source/component, not the mean of the bank RMSs.
The endpoints differ even though these counts are close. Along with the earlier
alias construction, this justifies resolving the continuous trajectory on a
finer grid. It does not identify which embedding component causes every change.

## Consequence

The native-field obstacle is cleared under the registered gate. Next refine
the complete oracle/tangent/four-blend calculation at2048/4096/8192while
retaining all original inputs. Native qualification alone does not qualify
the tangent, nor validate the first-order explanation at full model error.
Original large-bank Heun128SW2scores remain unchanged. The256-source paired
endpoint differences are neither SW2norFIDand establish no architecture win.

## Claim candidates

- Claim: all twelve fixed native fields pass the registered endpoint precision test.
  - Source evidence: summary.json, queue-finished.json and independent-COMPLETE.json.
  - Allowed wording: twelve finite-source endpoint checks pass, ten at4096and two at8192.
  - Forbidden stronger wording: all continuous trajectories or generation scores are exact.
  - Uncertainty: gate applies to all tested256sources, not unseen sources or training.
  - Next check: oracle/tangent/four-blend refinement.
  - Decision: keep.
- Claim: nearby sampling grids produce different endpoints at fixed weights.
  - Source evidence: all five contrasts per model and the grid-sensitivity figure.
  - Allowed wording: observed grid sensitivity motivates higher-resolution qualification.
  - Forbidden stronger wording: aliasing explains the earlier architecture ranking.
  - Uncertainty: source-coupled displacement need not imply distribution degradation.
  - Next check: qualified error-transport test, preserving ordinary controls.
  - Decision: keep.
''')
    stats = '# Statistics and numerical qualification\n\n'
    stats += 'Unit: the complete fixed256-source population per frozen checkpoint. The gate is deterministic, not a significance test. One training trajectory per model; no inference over training repeats.\n\n'
    stats += 'All-source RMS is sqrt(mean squared component difference). Maxima and componentwise gates are retained in summary.json. Bank standard deviation describes eight input groups. Conditional intervals enumerate all8^8ordered bank resamples through6435multisets; resample squared energies before taking the square root. They are descriptive input uncertainty, not evidence used to pass the component gate. No p-values or winner selection; no multiplicity-based superiority claim.\n\n'
    stats += '| Task/model | Contrast | Global RMS | Bank mean ± SD | Conditional95% RMS |\n|---|---|---:|---:|---|\n'
    for r in rows:
        for c in r['contrasts']:
            lo, hi = c['conditional95']
            stats += f"| {'Ring8' if r['dataset']==TASKS[0] else 'Product16'} / {LABEL[r['family']]} | {c['left']} vs{c['right']} | {c['rms']:.3e} | {c['bank_mean']:.3e} ± {c['bank_std']:.3e} | [{lo:.3e}, {hi:.3e}] |\n"
    (out/'stats-appendix.md').write_text(stats)
    (out/'figure-catalog.md').write_text('''# Figure catalog

## native-grid-convergence.png / .pdf
- Purpose: show why nearby/dyadic checks do not substitute for fine-grid qualification.
- Data: summary.json convergence_to_finest, recomputed from hash-bound native arrays.
- Caption: both tasks, all six models,256sources/model, categorical step axis; finest endpoint is an internal reference and its constructed zero is omitted; no error bars.
- Observation: fine-grid residuals shrink; two cases need8192under the component gate.
- Interpretation: native endpoints are ready for the refined sensitivity test.
- Check: do not interpret this as SW2, an analytic convergence proof or hidden-solver refinement.

## near-equal-work-grid-sensitivity.png / .pdf
- Purpose: isolate outer-time grid sensitivity at fixed checkpoints/inputs.
- Data: the1000/1024/1032contrasts in summary.json, all8banks retained.
- Caption: small dots are bank RMSs; diamonds pool squared error over all components/sources; no independent training repeats.
- Observation: nearby counts can differ appreciably; the finest reference is absent here.
- Interpretation: near-equal work does not establish equal continuous trajectories.
- Check: no accuracy ranking or distribution-distance claim follows from this figure.
''')
    dump(out/'COMPLETE.json', dict(passed=True, cases=12, qualified=source['qualified_cases'],
        input_complete_sha256=sha(root/'verified/COMPLETE.json'), source_sha256=sha(Path(__file__)),
        files={str(f.relative_to(out)): sha(f) for f in sorted(out.rglob('*')) if f.is_file()}))


if __name__ == '__main__':
    main()
