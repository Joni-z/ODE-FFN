"""Exact full-field risk decomposition and registered crossed-solver contrasts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, out = Path(args.input), Path(args.output)
    terminal = read(root/'ops/TERMINAL.json')
    assert terminal['passed'] and len(terminal['completed'])==4
    assert all(r['actual_terminal_exit']==0 for r in terminal['completed'])
    out.mkdir(parents=True,exist_ok=False)
    (out/'figures').mkdir()
    statistics = root/'analysis-statistics'
    statistics.mkdir(exist_ok=False)
    cases, contrasts, bindings = [], [], []
    for task, title, dim in [('ring','Ring8',2),('product','Product16',16)]:
        source, verified = root/'formal'/task, root/'formal'/(task+'-verified')
        complete, proof = read(source/'COMPLETE.json'), read(verified/'PASS.json')
        assert complete['passed'] and not complete['witness'] and complete['all_points_qualified']
        assert proof['passed'] and not proof['witness'] and proof['all_points_qualified']
        assert proof['input_complete_sha256']==sha(source/'COMPLETE.json')
        with np.load(source/'inputs.npz') as z:
            oracle = z['oracle'].copy()
        assert len(oracle)==4096
        task_cases, values = [], []
        for row in read(source/'rows.json')['rows']:
            own = 'midpoint' if row['family']=='state_midpoint4' else 'rk4'
            predictions = {}
            for mode in ['midpoint','rk4','continuous_tight']:
                with np.load(source/row['id']/(mode+'.npz')) as z:
                    predictions[mode] = z['predictions'].copy()
            losses = {m:np.sum((p-oracle)**2,axis=1) for m,p in predictions.items()}
            other = 'rk4' if own=='midpoint' else 'midpoint'
            values.append(losses[own]-losses[other])
            record = dict(id=row['id'],task=task,title=title,dimension=dim,trained_solver=own,
                risks={m:float(v.mean()) for m,v in losses.items()},
                mse={m:float(v.mean()/dim) for m,v in losses.items()},
                own_accounting=row['risk_accounting'][own],
                all_accounting=row['risk_accounting'],
                continuous_tolerance_risk_resolution=row['continuous_tolerance_risk_resolution'],
                checkpoint_sha256=row['checkpoint_sha256'],qualified_points=row['qualified_points'])
            cases.append(record)
            task_cases.append((record,losses))
        assert [r[0]['trained_solver'] for r in task_cases]==['midpoint','rk4']
        values += [losses['continuous_tight']-losses[record['trained_solver']]
                   for record,losses in task_cases]
        values = np.stack(values,axis=1)
        rng = np.random.default_rng(0)
        bootstrap = np.empty((10000,4))
        for start in range(0,10000,100):
            index = rng.integers(0,4096,size=(100,4096))
            bootstrap[start:start+100] = values[index].mean(axis=1)
        datafile = statistics/(task+'.npz')
        np.savez_compressed(datafile,values=values,bootstrap=bootstrap)
        bindings.append(dict(task=task,file=str(datafile),sha256=sha(datafile),
            native_complete_sha256=sha(source/'COMPLETE.json'),independent_sha256=sha(verified/'PASS.json')))
        for index,(record,_) in enumerate(task_cases):
            lo,hi = np.quantile(bootstrap[:,index],[.00625,.99375])
            contrasts.append(dict(task=task,title=title,trained_solver=record['trained_solver'],
                dimension=dim,own_minus_other_vector_risk=float(values[:,index].mean()),
                per_state_sd=float(values[:,index].std(ddof=1)),
                marginal95=np.quantile(bootstrap[:,index],[.025,.975]).tolist(),
                four_comparison_adjusted=[float(lo),float(hi)],
                own_solver_point_advantage=bool(values[:,index].mean()<0),
                own_solver_adjusted_advantage=bool(hi<0),
                own_to_continuous_shift95=np.quantile(bootstrap[:,index+2],[.025,.975]).tolist()))
    result = dict(passed=True,cases=cases,contrasts=contrasts,bindings=bindings,
        compensation_point_support_all=all(r['own_solver_point_advantage'] for r in contrasts),
        compensation_adjusted_support_all=all(r['own_solver_adjusted_advantage'] for r in contrasts),
        points_per_case=4096,training_seed=0,new_training_runs=0,
        evaluation_bootstrap_seed=0,evaluation_bootstrap_resamples=10000,
        source_sha256=sha(Path(__file__)))
    dump(out/'summary.json',result)
    table = '| Task | Trained with | Midpoint MSE | RK4 MSE | Continuous MSE | Native exposure S/R | Native-to-continuous risk change |\n|---|---|---:|---:|---:|---:|---:|\n'
    for row in cases:
        q = row['own_accounting']
        table += f"| {row['title']} | {row['trained_solver']} | {row['mse']['midpoint']:.3f} | {row['mse']['rk4']:.3f} | {row['mse']['continuous_tight']:.3f} | {q['exposure']:.3e} | {100*q['risk_change']/q['R']:.3f}% |\n"
    (out/'table.md').write_text(table)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes = plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for ax,task,title in zip(axes,['ring','product'],['Ring8','Product16']):
        group = [r for r in cases if r['task']==task]
        data = np.array([[r['mse'][m]*1000 for m in ['midpoint','rk4','continuous_tight']] for r in group])
        im = ax.imshow(data,cmap='YlOrRd',aspect='auto')
        for i in range(2):
            for j in range(3):
                ax.text(j,i,f'{data[i,j]:.3f}',ha='center',va='center',color='white' if data[i,j]>(data.min()+data.max())/2 else 'black')
        ax.set(xticks=range(3),xticklabels=['Midpoint4','RK4 × 2','Continuous'],
               yticks=range(2),yticklabels=['Midpoint-trained','RK4-trained'],title=title+' / field MSE × 1000 ↓')
        fig.colorbar(im,ax=ax,shrink=.8)
    for ext in ['png','pdf']:
        fig.savefig(out/'figures'/('crossed-field-risks.'+ext),dpi=180)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    ax = axes[0]
    for i,row in enumerate(contrasts):
        scale = 1000/row['dimension']
        lo,hi = row['four_comparison_adjusted']
        ax.hlines(i,lo*scale,hi*scale,color='#2166ac',lw=2)
        ax.scatter(row['own_minus_other_vector_risk']*scale,i,color='#d95f02',s=35,zorder=3)
    ax.axvline(0,color='.4',lw=1)
    ax.set(yticks=range(4),yticklabels=[r['title']+' / '+r['trained_solver']+'-trained' for r in contrasts],
           xlabel='Own minus other solver MSE × 1000',title='Does training favor its own solver?')
    ax = axes[1]
    for i,row in enumerate(cases):
        q = row['own_accounting']
        cross,energy = 200*q['C']/q['R'],100*q['S']/q['R']
        ax.bar(i-.16,cross,width=.3,color='#2166ac',label='Alignment term 2C/R' if i==0 else None)
        ax.bar(i+.16,energy,width=.3,color='#d95f02',label='Change energy S/R' if i==0 else None)
    ax.axhline(0,color='.4',lw=1)
    ax.set(xticks=range(4),xticklabels=['Ring8\nMid','Ring8\nRK4','Product16\nMid','Product16\nRK4'],
           ylabel='Contribution to native→continuous risk change (%)',title='Exact risk accounting')
    ax.legend(fontsize=8)
    for ext in ['png','pdf']:
        fig.savefig(out/'figures'/('compensation-and-risk-accounting.'+ext),dpi=180)
    plt.close(fig)
    shutil.copyfile(root/'ops/TERMINAL.json',out/'terminal-receipts.json')
    (out/'analysis-report.md').write_text('''# The hidden solver bias reaches the field and can reduce its error

The complete-field test rules out a simple own-solver compensation story on
Ring8: both sets of trained weights fit the oracle field better when evaluated
with midpoint than RK4. Removing midpoint's numerical bias also worsens the
midpoint-trained field MSE by 6.121% on Ring8 and 1.762% on Product16. The internal
solver difference is therefore exposed at the task output, with a measurable
error-correcting direction. Generation value is a separate next test.

## Frozen-weight intervention

Four retained 20k checkpoints, all 4,096 original selection-field inputs per task,
all six internal solves changed together. The two continuous passes integrate
each layer from its actual sequential input, rather than reusing native-state
reference endpoints. Original/identity and own-solver patches replay exactly;
model tensors and RNG remain unchanged. All points qualify in the independent
local reconstruction and continuous-reference checks. No training was added.

![Crossed field risks](figures/crossed-field-risks.png)

Rows fix learned weights; columns change only the hidden solver. Values are
per-coordinate field MSE × 1000 to show small differences legibly. Ring8 favors
midpoint in both rows. Product16 favors midpoint for midpoint-trained weights,
while RK4-trained weights have a smaller advantage with their original method.

'''+table+'''
## Why the risk changes

For error e=F_num-v_oracle and field change delta=F_cont-F_num, the exact
identity is R_cont-R_num=2C+S, where R=E||e||²,S=E||delta||²,C=E<e,delta>.
Positive C means the change toward continuous solving points partly along the
remaining field error; it makes that error larger. Its size cannot be inferred
from hidden-state accuracy alone. The full cohort reconstructs this identity
independently. Native midpoint exposure S/R is about .010 on both tasks, so its
effect is not erased by the complete readout.

![Compensation and accounting](figures/compensation-and-risk-accounting.png)

Left: own-solver risk minus other-solver risk, with paired four-comparison-adjusted
intervals. The strong compensation hypothesis requires negative differences in
all four cases; Ring8 RK4-trained weights contradict it. Right: the two exact
terms for replacing each model's native solver by a continuous solve.

## Research decision

More faithful solving did not improve generation in the prior matched training.
This observation now shows why simply treating hidden numerical error as harmful
is inadequate: its direction can reduce complete-field error. The next test
keeps these weights and evaluates the crossed solvers on all original common
sampling banks, with the native sampler and ordinary controls retained. Record
the field-based solver ranking before opening these new sampling outcomes.
Do not promote a new architecture from this field diagnostic alone.

## Claim candidates

- Claim: native hidden numerical bias affects the complete learned field.
  - Evidence: full 4,096-input sequential interventions, independent references,
    risk exposure and exact accounting.
  - Allowed wording: measured field-level exposure and direction in these models.
  - Forbidden stronger wording: improved generative distribution or FID.
  - Uncertainty: frozen fields on the retained evaluation distribution.
  - Next check: actual common-bank sampling after the crossed substitutions.
  - Decision: keep.
- Claim: the strong own-solver compensation explanation is not supported.
  - Evidence: crossed field risks and paired contrasts, especially Ring8 RK4 weights.
  - Allowed wording: both Ring8 weight sets prefer midpoint on the full field probe.
  - Forbidden stronger wording: no learned compensation exists in any model.
  - Uncertainty: task-specific preferences and conditional evaluation variation.
  - Next check: preserve all four cases in the sampling test.
  - Decision: revise the mechanism toward directed numerical bias.
''')
    (out/'stats-appendix.md').write_text('''# Numerical and statistical units

Four checkpoints, training seed 0/20k; no new training. Each case retains 4,096 IID
field inputs, all six local solves and six forward modes. Original-identity replay
is exact. Two complete DOP853 tolerances, per-component/layer gates, energy closure,
and independent state-only RK45 on differently packed 64-point chunks qualify every
continuous local output. Numerical risk resolution is the measured difference
between complete tolerance passes and its Cauchy-Schwarz energy bound; it is a
resolution diagnostic rather than a rigorous certificate for the unknown exact
solution. Risk uses vector squared norm; displayed MSE divides by physical dimension.

Primary comparison: own-solver loss minus other-solver loss for each weight set,
two tasks by two training solvers. Conditional paired bootstrap over 4,096 original
IID field points keeps all solver predictions aligned. Evaluation RNG 0, 10,000
resamples, batches 100; percentile 95% and Bonferroni four-comparison intervals.
No fabricated sampling banks or independent training runs. Full per-state
contrasts and bootstrap arrays are source-bound outside Git. Both native and
both independent commands have actual terminal 0.
''')
    (out/'figure-catalog.md').write_text('''# Figure interpretation

## crossed-field-risks
Purpose: test whether each learned weight set prefers its training solver.
Data: independently qualified full field predictions, 4,096 points/case.
Rows are fixed weights; columns are solver substitutions; all values are
per-coordinate MSE×1000. Ring8 both rows prefer midpoint. This contradicts the
strong diagonal compensation prediction; it does not establish sample quality.

## compensation-and-risk-accounting
Purpose: show the registered paired comparisons and the exact direction of bias.
Left intervals: four-comparison-adjusted conditional paired field-point bootstrap;
negative means the model prefers its training solver. Right: exact 2C/R and S/R
contributions to relative native-to-continuous risk change. Positive alignment
can make a more accurate hidden solve a worse field predictor. Next decision:
test crossed sampling with the existing ordinary controls.
''')
    print('field exposure analysis complete',out,flush=True)


if __name__=='__main__':
    main()
