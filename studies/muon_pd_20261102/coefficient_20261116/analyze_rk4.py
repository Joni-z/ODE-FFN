"""Join qualified hidden fidelity, verified samples and fixed matched controls."""
import argparse
import itertools
import json
import math
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TASKS = ('gmm_ring8_2d', 'rotated_product_bimodal_16d')
LABEL = {'state_midpoint4':'Midpoint / 4 steps', 'state_rk4_2':'RK4 / 2 steps',
         'free_isru':'Free ISRU', 'swiglu64':'SwiGLU64 / Muon', 'swiglu192':'SwiGLU192 / AdamW'}


def read(path: Path):
    return json.loads(path.read_text())


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def compositions() -> tuple:
    counts, weights = [], []
    for sample in itertools.combinations_with_replacement(range(8), 8):
        c = np.bincount(sample, minlength=8)
        counts.append(c)
        weights.append(math.factorial(8)//math.prod(math.factorial(int(v)) for v in c))
    counts, weights = np.asarray(counts), np.asarray(weights, dtype=np.int64)
    assert len(counts) == 6435 and weights.sum() == 8**8
    return counts, weights


def intervals(values: np.ndarray, weights: np.ndarray) -> dict:
    order = np.argsort(values)
    cdf = np.cumsum(weights[order])
    def at(q: float) -> float:
        return float(values[order[np.searchsorted(cdf, math.ceil(q*8**8))]])
    return {name:[at(q), at(1-q)] for name,q in
        [('conditional95', .025), ('nominal_four_adjusted', .05/8), ('six_contrast_sensitivity', .05/12)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.output)
    native = read(root/'ops/TERMINAL.json')
    metrics = read(root/'metrics-verified/PASS.json')
    terminal = read(root/'fidelity-formal/terminal-receipts.json')
    cost = read(root/'cost.json')
    assert native['passed'] and metrics['passed'] and terminal['passed'] and cost['passed']
    assert len(terminal['commands']) == 8 and all(r['actual_terminal_exit'] == 0 for r in terminal['commands'])
    counts, weights = compositions()
    out.mkdir(parents=True, exist_ok=False)
    figures = out/'figures'
    figures.mkdir()
    fidelity = []
    for short, task in zip(['ring', 'product'], TASKS):
        for solver, family in [('midpoint','state_midpoint4'), ('rk4','state_rk4_2')]:
            folder = root/'fidelity-formal'/(short+'-'+solver)
            row = read(folder/'rows.json')
            verified = read(root/'fidelity-formal'/(short+'-'+solver+'-verified')/'PASS.json')
            assert not row['witness'] and row['step'] == 20000
            assert row['all_points_qualified'] and verified['all_points_qualified']
            assert len(row['layers']) == 6
            with np.load(folder/'inputs.npz') as bundle:
                bank = bundle['bank'].copy()
            assert len(bank) == 80 and np.array_equal(np.bincount(bank), np.repeat(10, 8))
            error, displacement = np.zeros((3, 8)), np.zeros(8)
            for layer in row['layers']:
                with np.load(folder/layer['file']) as bundle:
                    for b in range(8):
                        error[:, b] += bundle['error_squared'][:, bank == b].sum(axis=1)
                        displacement[b] += bundle['displacement_squared'][bank == b].sum()
            assert np.all(displacement > 0)
            matching = next(r for r in metrics['rows'] if r['dataset'] == task and r['family'] == family)
            assert matching['checkpoint_sha256'] == row['checkpoint_sha256']
            fidelity.append(dict(dataset=task, family=family, error_bank=error[0].tolist(),
                displacement_bank=displacement.tolist(), bank_rho=(error[0]/displacement).tolist(),
                rho=float(error[0].sum()/displacement.sum()), total_squared_error=float(error[0].sum()),
                total_displacement_energy=float(displacement.sum()),
                error_rms=float(np.sqrt(error[0].sum()/480)),
                refined_rho=(error.sum(axis=1)/displacement.sum()).tolist(),
                layers=row['layers'], checkpoint_sha256=row['checkpoint_sha256']))
    contrasts = []
    for task in TASKS:
        for split in ['selection', 'confirmation']:
            candidate = next(r for r in metrics['rows'] if r['dataset'] == task and r['split'] == split and r['family'] == 'state_rk4_2')
            reference = next(r for r in metrics['rows'] if r['dataset'] == task and r['split'] == split and r['family'] == 'state_midpoint4')
            delta = np.asarray(candidate['bank_sw2'])-reference['bank_sw2']
            contrasts.append(dict(dataset=task, metric='bank_sw2', split=split,
                bank_differences=delta.tolist(), point_difference=float(delta.mean()),
                sd_difference=float(delta.std(ddof=1)), pooled_sw2_difference=candidate['pooled_sw2']-reference['pooled_sw2'],
                better_banks=int((delta < 0).sum()), intervals=intervals(counts@delta/8, weights),
                quality_point_support=bool(delta.mean() < 0 and candidate['pooled_sw2'] < reference['pooled_sw2'])))
        candidate = next(r for r in fidelity if r['dataset'] == task and r['family'] == 'state_rk4_2')
        reference = next(r for r in fidelity if r['dataset'] == task and r['family'] == 'state_midpoint4')
        delta = np.asarray(candidate['bank_rho'])-reference['bank_rho']
        bootstrap = (counts@candidate['error_bank'])/(counts@candidate['displacement_bank'])
        bootstrap -= (counts@reference['error_bank'])/(counts@reference['displacement_bank'])
        contrasts.append(dict(dataset=task, metric='hidden_rho', split='fixed_interpolation_inputs',
            candidate_error_bank=candidate['error_bank'], reference_error_bank=reference['error_bank'],
            candidate_displacement_bank=candidate['displacement_bank'], reference_displacement_bank=reference['displacement_bank'],
            bank_differences=delta.tolist(), point_difference=candidate['rho']-reference['rho'],
            mean_bank_difference=float(delta.mean()), better_banks=int((delta < 0).sum()),
            reduction_factor=reference['rho']/candidate['rho'], intervals=intervals(bootstrap, weights),
            fidelity_point_support=bool(candidate['rho'] < reference['rho'])))
    result = dict(passed=True, metrics=metrics['rows'], fidelity=fidelity, contrasts=contrasts,
        quality_support_all_splits_tasks=all(r['quality_point_support'] for r in contrasts if r['metric']=='bank_sw2'),
        fidelity_support_both_tasks=all(r['fidelity_point_support'] for r in contrasts if r['metric']=='hidden_rho'),
        cost=cost, ordered_resamples=8**8, multiset_groups=6435,
        training_seed=0, training_steps=20000, numerical_verification_max_error=metrics['maximum_numpy_difference'])
    dump(out/'summary.json', result)
    for source, name in [(root/'ops/TERMINAL.json','training-terminal.json'),
                         (root/'fidelity-formal/terminal-receipts.json','observer-terminal.json'),
                         (root/'metrics-verified/PASS.json','metrics-verification.json'),
                         (root/'fidelity-readiness/READY.json','observer-readiness.json')]:
        shutil.copyfile(source, out/name)
    table = '| Task | Model | Field MSE | Pooled SW2 | Bank SW2 mean ± SD | Hidden rho |\n|---|---|---:|---:|---:|---:|\n'
    for task, title in zip(TASKS, ['Ring8', 'Product16']):
        for row in [r for r in metrics['rows'] if r['dataset']==task and r['split']=='confirmation']:
            hidden = next((f for f in fidelity if f['dataset']==task and f['family']==row['family']), None)
            rho = f"{hidden['rho']:.3e}" if hidden else '—'
            table += f"| {title} | {LABEL[row['family']]} | {row['field_mse']:.3f} | {row['pooled_sw2']:.3f} | {row['sw2_mean']:.3f} ± {row['sw2_sd']:.3f} | {rho} |\n"
    (out/'table.md').write_text(table)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for column, (task, title) in enumerate(zip(TASKS, ['Ring8', 'Product16'])):
        group = [r for r in metrics['rows'] if r['dataset']==task and r['split']=='confirmation']
        group.sort(key=lambda r: ['state_midpoint4','state_rk4_2','free_isru','swiglu64','swiglu192'].index(r['family']))
        ax = axes[0, column]
        ax.bar(range(4), [r['pooled_sw2'] for r in group], color=['#2166ac','#d95f02','#7570b3','#444444'])
        ax.set(xticks=range(4), xticklabels=['Midpoint4','RK4 × 2','Free ISRU','SwiGLU'],
               ylabel='Pooled target SW2 ↓', title=title+' / generation')
        ax = axes[1, column]
        group = [r for r in fidelity if r['dataset']==task]
        ax.plot(range(2), [r['rho'] for r in group], color='.7', lw=1)
        ax.scatter(range(2), [r['rho'] for r in group], color=['#2166ac','#d95f02'], s=55, zorder=3)
        for i, row in enumerate(group):
            ax.annotate(f"{row['rho']:.3e}", (i,row['rho']), xytext=(0,9),
                        textcoords='offset points', ha='center', fontsize=9)
        ax.set_ylim(min(r['rho'] for r in fidelity)/2, max(r['rho'] for r in fidelity)*2)
        ax.set(xticks=range(2), xticklabels=['Midpoint / 4 steps','RK4 / 2 steps'],
               yscale='log', ylabel='Normalized hidden error rho ↓', title=title+' / continuous fidelity')
    for ext in ['png','pdf']:
        fig.savefig(figures/('fidelity-and-generation.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for column, (task, title) in enumerate(zip(TASKS, ['Ring8', 'Product16'])):
        controls = read(Path(__file__).resolve().parents[3]/'docs/research-assessments/2026-11-16-ode-mechanism/rk4-control-bindings.json')
        old = next(r for r in controls['references'] if r['dataset']==task and r['family']=='state_midpoint4')
        new = next(r for r in native['cases'] if r['dataset']==task)
        for row, color, label in [(old,'#2166ac','Midpoint4'),(new,'#d95f02','RK4 × 2')]:
            values = [json.loads(line) for line in (Path(row['folder'])/'metrics.jsonl').read_text().splitlines()]
            values = [r for r in values if r['step'] > 0]
            axes[0,column].plot([r['step'] for r in values], [r['validation_oracle_mse'] for r in values], color=color,label=label)
        axes[0,column].set(title=title+' / native training history',xlabel='Training updates',ylabel='Validation oracle MSE ↓',yscale='log')
        axes[0,column].legend()
        row = next(r for r in contrasts if r['dataset']==task and r['metric']=='bank_sw2' and r['split']=='confirmation')
        lo,hi = row['intervals']['nominal_four_adjusted']
        ax = axes[1,column]
        ax.scatter(np.asarray(row['bank_differences'])*1000, np.arange(8), color='.6', label='Paired banks')
        ax.hlines(8.3,lo*1000,hi*1000,color='#d95f02',lw=3)
        ax.scatter(row['point_difference']*1000,8.3,color='#d95f02',label='Mean / adjusted interval')
        ax.axvline(0,color='.3',lw=1)
        ax.set(title=title+' / paired confirmation banks',xlabel='RK4 minus midpoint SW2 × 1000',yticks=range(9),yticklabels=[str(i) for i in range(8)]+['Mean'])
    for ext in ['png','pdf']:
        fig.savefig(figures/('training-and-paired-sampling.'+ext), dpi=180)
    plt.close(fig)
    (out/'analysis-report.md').write_text('''# More faithful hidden ODE solves do not improve this generator comparison

At the same parameter count and dense matrix work, RK4 reduces normalized hidden
solver error by about 46x on Ring8 and 22x on Product16. Target SW2 does not improve
on the confirmation split of either task. The registered two-task quality
prediction fails while the hidden-fidelity prediction succeeds.

## What changed

Both models solve y'=P ISRU(y), with the same trainable tensors, initialization,
training data stream, AdamW 3e-4 and 20k updates. Four explicit midpoint steps are
replaced by two classical RK4 steps. Both use eight P applications per local solve.
The numerical RK4 step is copied from the attributed torchdiffeq source. This is
state-updated synthetic feature evolution, not image-model frozen-D Taylor K8.

![Fidelity and target quality](figures/fidelity-and-generation.png)

The lower row shows successful numerical intervention; the upper row shows its
absence of a generation gain. Ordinary controls stay in the comparison. On
Product16 the native SwiGLU configuration remains stronger than both ODE variants.

'''+table+'''
Both selection/confirmation outcomes, every bank and all hidden layers remain
in summary.json. Ring8 selection pooled SW2 decreases slightly, but confirmation
increases; Product16 increases on both splits. These differences do not establish
a useful RK4 candidate. No checkpoint or sampling bank is chosen after evaluation.

![Training and paired effects](figures/training-and-paired-sampling.png)

Training histories use the unchanged native validation probe. Bottom panels
show all eight confirmation-bank differences and the registered adjusted interval;
negative favors RK4. The tiny quality shifts contrast with the large reduction
in hidden solver error. This rejects continuous-solve accuracy as a sufficient
explanation for this architecture's generation performance at this budget.

## Interpretation and next question

The earlier modified-equation calculation successfully predicted the midpoint
bias. Removing much of that bias during training is now shown not to buy better
samples here. Mathematical fidelity is measurable, but its usefulness depends
on how the complete network exposes those hidden changes to the predicted field.
The next bounded diagnostic should measure that exposure directly: compare
complete-field changes from faithful hidden solves with the actual field-error
scale, using frozen learned weights and all local solves. It should distinguish
weak output sensitivity from learned compensation before another architecture.

## Claim candidates

- Claim: equal-matrix-work RK4 improves hidden continuous fidelity on both tasks.
  - Evidence: all 1,920 states, both reference checks, eight actual observer exits 0.
  - Allowed wording: the recorded same-budget solver comparison is more faithful.
  - Forbidden stronger wording: ODE models generally outperform ordinary FFNs.
  - Uncertainty: learned endpoints and these task/operator configurations.
  - Next check: full-field exposure and learned compensation.
  - Decision: keep.
- Claim: better hidden accuracy is insufficient for improved target SW2 here.
  - Evidence: both complete training trajectories and common-bank comparisons;
    independent NumPy reconstruction includes both ordinary controls.
  - Allowed wording: the registered cross-task quality prediction fails.
  - Forbidden stronger wording: every possible continuous-depth design is useless.
  - Uncertainty: this isolates solver faithfulness within the fixed hidden field.
  - Next check: complete-field sensitivity at the retained learned weights.
  - Decision: keep.
''')
    (out/'stats-appendix.md').write_text('''# Analysis units and exact conditional intervals

Training seed 0, 20k updates, unchanged recipe. Quality: eight common 4096-source
banks per split, Heun 128, 256 projection directions; pooled score uses 32,768 samples.
All16 model/split score rows are independently reconstructed in NumPy. Fidelity:
four cases, six local solves and 80 fixed inputs per case; all 1,920 points retained
and qualified by two DOP853 tolerances, energy closure and independent per-input RK45.
Pooled rho is total squared hidden error divided by total continuous displacement.

Resampling unit is the original bank, retaining layers/times/pairs together.
Enumerate 6,435 multisets with exact multinomial weights for all 8^8 ordered bank
resamples. Quality intervals concern mean paired-bank SW2 differences, not pooled
SW2. Fidelity intervals recompute the difference of pooled energy ratios.
Report marginal 95%, registered nominal four-comparison intervals and a conservative
six-comparison sensitivity including both quality splits. These are conditional
evaluation-bank intervals for the retained weights. Do not interpret banks as
independent training runs. No arbitrary performance margin or checkpoint selection.

Cost uses the same process, alternating method order, 10 warmups and 100 repeats,
batch 512 and one CPU thread. Dense MACs and parameters match; weighted additions
differ. Native launch durations include polling granularity and final evaluation,
so they are recorded rather than used as matched throughput estimates.
''')
    (out/'figure-catalog.md').write_text('''# Figure interpretation

## fidelity-and-generation
Purpose: place the successful numerical intervention next to its task outcome.
Sources: independently verified confirmation samples and qualified hidden arrays.
Top: pooled target SW2 for both solvers and ordinary controls, no error bars on the
pooled statistic. Bottom: pooled normalized hidden error, logarithmic scale.
Observe: hidden accuracy improves strongly; generated quality does not.
Implication: accurate hidden solving alone is not sufficient for task improvement.

## training-and-paired-sampling
Purpose: show actual training trajectories and the size of paired sample changes.
Sources: immutable native metrics and the eight common confirmation banks.
Top: native validation oracle MSE history. Bottom: individual bankSW2 differences
and nominal four-comparison-adjusted exact conditional-bootstrap interval of
their mean. Negative favors RK4. Bank variation is not training-repeat variation.
Implication: inspect full-field exposure rather than raising solver order again.
''')
    print('analysis complete', out, flush=True)


if __name__ == '__main__':
    main()
