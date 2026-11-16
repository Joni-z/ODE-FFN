"""Finite-cohort numerical qualification; no performance inference."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from geometry import read, dump, sha

LABELS = {'state_midpoint4': 'Shared P', 'palindrome_operator_midpoint4': 'Palindrome',
          'alternating_operator_midpoint4': 'Alternating'}
LEVELS = (16, 32, 64, 128, 256)


def worst(level: dict, key: str) -> float:
    return max(layer[key] for layer in level['layers'])


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
        assert complete['cases'] == proof['cases'] == 3 and complete['inputs_per_case'] == 80 and complete['layers_per_case'] == 6
        assert sha(verified/'summary.json') == proof['summary_sha256']
        assert summary['input_complete_sha256'] == sha(native/'COMPLETE.json')
        rows += summary['rows']
        max_error = max(max_error, summary['max_reconstruction_error'])
        bindings.append(dict(task=task, native_complete_sha256=sha(native/'COMPLETE.json'),
            verified_complete_sha256=sha(verified/'COMPLETE.json'), verified_summary_sha256=sha(verified/'summary.json')))
    assert len(rows) == 6
    qualified = [level for level in LEVELS if all(next(item for item in row['levels']
                 if item['steps'] == level)['qualified'] for row in rows)]
    selected = qualified[0] if qualified else None
    decision = ('Use the common resolution only for a separately qualified on-trajectory sampling witness.'
                if selected else 'No common surrogate qualifies through256; do not launch sampling under this protocol.')
    out.mkdir(parents=True, exist_ok=False)
    (out/'figures').mkdir()
    dump(out/'summary.json', dict(rows=rows, bindings=bindings, common_qualified_levels=qualified,
        selected_resolution=selected, max_reconstruction_error=max_error, decision=decision))
    shutil.copyfile(root/'terminal-receipts.json', out/'terminal-receipts.json')
    shutil.copyfile(root/'readiness/READY.json', out/'readiness.json')
    table = '| Task | Model | Steps | Local error / limit | Output error / limit | Largest F error | Qualified |\n|---|---|---:|---:|---:|---:|---|\n'
    for row in rows:
        for level in row['levels']:
            f_error = max([level['bank_fraction_max_error'], level['model_fraction_max_error']]+
                          [item['fraction_max_error'] for item in level['layers']])
            table += (f"| {'Ring8' if row['dataset']=='gmm_ring8_2d' else 'Product16'} | {LABELS[row['family']]} | {level['steps']} | "
                      f"{worst(level, 'local_max_limit_ratio'):.3f} | {worst(level, 'output_max_limit_ratio'):.3f} | "
                      f"{f_error:.3e} | {'yes' if level['qualified'] else 'no'} |\n")
    (out/'table.md').write_text(table)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    colors = ['#2166ac', '#d95f02', '#1b9e77']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    for ax, task, title in zip(axes, ['gmm_ring8_2d', 'rotated_product_bimodal_16d'], ['Ring8', 'Product16']):
        group = [row for row in rows if row['dataset'] == task]
        for color, row in zip(colors, group):
            for key, style, suffix in [('local_max_limit_ratio', '--', 'local'), ('output_max_limit_ratio', '-', 'output')]:
                ax.plot(LEVELS, [worst(level, key) for level in row['levels']], style, marker='o',
                        ms=4, color=color, label=LABELS[row['family']]+' '+suffix)
        ax.axhline(1, color='black', lw=1, label='Qualification limit')
        ax.set(xscale='log', yscale='log', title=title, xlabel='Native midpoint substeps', ylabel='Worst pointwise error / allowed error')
        ax.set_xticks(LEVELS, [str(level) for level in LEVELS])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.legend(fontsize=7, ncol=2)
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('pointwise-reference-qualification.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    for ax, task, title in zip(axes, ['gmm_ring8_2d', 'rotated_product_bimodal_16d'], ['Ring8', 'Product16']):
        for color, row in zip(colors, [row for row in rows if row['dataset'] == task]):
            errors = [max([level['bank_fraction_max_error'], level['model_fraction_max_error']]+
                      [item['fraction_max_error'] for item in level['layers']]) for level in row['levels']]
            ax.plot(LEVELS, errors, marker='o', color=color, label=LABELS[row['family']])
        ax.axhline(.0005, color='black', lw=1, label='Half a displayed decimal unit')
        ax.set(xscale='log', yscale='log', title=title, xlabel='Native midpoint substeps', ylabel='Largest absolute reproduction-fraction error')
        ax.set_xticks(LEVELS, [str(level) for level in LEVELS])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.legend(fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(out/'figures'/('fraction-reference-qualification.'+ext), dpi=180)
    plt.close(fig)
    outcome = f'Common qualified levels: {qualified}. Selected resolution: {selected}.'
    (out/'analysis-report.md').write_text(f'''# Can native refinement replace the continuous reference?

**{outcome}** {decision}

The complete comparison retains six fixed20kseed0models,80registered inputs
per model and all six separate local-layer interventions. Each of five native
resolutions is compared with the already qualified DOP853local and full-output
reference/normal/tangential/exact-potential-ray interventions. There is no new
training or sampling-performance comparison.

## Pointwise approximation

![Pointwise numerical limits](figures/pointwise-reference-qualification.png)

Every curve reports the worst of all input points, modes and layers for its
model. A value below1passes the registered bound. Local and output errors are
scaled by the original intervention response, not by raw feature magnitude.
The curves determine whether numerical replacement is admissible on this cohort.

## Reproduction-fraction stability

![Fraction stability](figures/fraction-reference-qualification.png)

The ordinate is the maximum absolute Fdifference over layer, input-bank and
whole-model aggregates. The fixed bound0.0005is half the three-decimal display
unit. Passing this aggregate check alone does not override pointwise failures.

{table}

Both native datasets and both independent verifiers exit0. Independent endpoint
and candidate reconstruction agrees within{max_error:.3e}. Source arrays,
checkpoint identities, native roundtrips and unchanged parameters/buffers are
retained. Exact one-layer matrix work is(2L+40)/48relative to the original
six-layer network for direct replacement by a refined endpoint; runtime and
every failed level remain in summary.json. Component corrections additionally
need the coarse endpoint, so their matrix work is(48+2L)/48before hook and
projection overhead. The producer's one_layer_matrix_work_multiplier records
only the ideal direct-endpoint replacement, not measured intervention speed.

## Decision

{decision} This finite interpolation-state qualification does not establish
accuracy on generated trajectories. That requires a separate fixed-layer
resolution check before full-bank sampling. No final-distribution improvement,
ODE-specific advantage or new image architecture is inferred here.

## Claim candidates

- Claim: the listed native resolutions meet or fail the fixed reference limits.
  - Source evidence: all2880states, five resolutions, four intervention modes;
    independent NumPy reconstruction and complete layer/bank/model aggregates.
  - Allowed wording: numerical qualification on this complete registered cohort.
  - Forbidden stronger wording: generated trajectories are qualified or improve.
  - Uncertainty: new trajectory states can have different stiffness and response.
  - Next check: fixed-layer on-trajectory resolution and original-bank sampling.
  - Decision: retain the numerical outcome without changing thresholds.
''')
    (out/'stats-appendix.md').write_text('''# Numerical comparison units and limits

This is a deterministic approximation qualification, not a population
performance comparison. Six fixed trained checkpoints are retained, with80
registered inputs each and six layers; all2880local states are evaluated at
all five resolutions and four nonidentity modes. No points are removed.
Maxima and violation counts are the relevant statistics because every point
must meet the registered gate. Means, p-values and confidence intervals cannot
replace that universal finite-cohort check and are not used for admission.

Bounds are fixed in hidden-surrogate-protocol.md: local1e-9+1e-3times native
refinement norm; output1e-10+1e-3times full-reference response norm; Ferror
<=0.0005at each layer, bank and whole model. Every failing level is retained.
Fuses each method's own reference displacement and is not generation quality.
Eight input banks are repeated inputs at fixed weights, not independent
training replicates. No training-seed inference is made.

The common resolution is selected across every registered model/layer, rather
than by a favorable score. Zero response denominators remain unresolved.
Native Torchand a separate NumPymidpoint implementation must agree within
1e-11absolute/relative. The source DOP853references have their own prior,
unchanged two-tolerance qualification. No test of improved sampling occurs.
''')
    (out/'figure-catalog.md').write_text('''# Figure catalog

## pointwise-reference-qualification
Purpose: determine which fixed native resolutions satisfy all pointwise gates.
Data: independent formal summaries, worst local/output limit ratio over each
model's80inputs, six layers and four nonidentity modes. No error bars: these
are observed finite-cohort maxima, not estimates of population uncertainty.
Caption must identify the fixed floor/relative scale and threshold1.
Notice whether every curve clears1at the same level; one passing model does
not qualify the common surrogate. Do not infer on-trajectory accuracy.

## fraction-reference-qualification
Purpose: verify downstream interpretation is retained at the reporting scale.
Data: maximum absolute Fchange over all layers, banks and complete model.
Caption must distinguish numerical Ferror from a change in generation quality
and state the0.0005limit. No inferential intervals apply to this deterministic
gate. Both this and the pointwise gate must pass; aggregate cancellation alone
cannot qualify a level. The result determines the next numerical witness.
''')
    print(outcome, decision, flush=True)


if __name__ == '__main__':
    main()
