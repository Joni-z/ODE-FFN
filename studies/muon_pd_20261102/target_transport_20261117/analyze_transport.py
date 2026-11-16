"""Complete-cohort target-transport analysis; no checkpoint or bank selection."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compositions():
    # Same exhaustive multinomial enumeration as coefficient_20261116/analyze_rk4.py.
    counts, weights = [], []
    for sample in itertools.combinations_with_replacement(range(8),8):
        count = np.bincount(sample,minlength=8)
        counts.append(count)
        weights.append(math.factorial(8)//math.prod(math.factorial(int(v)) for v in count))
    counts,weights = np.asarray(counts),np.asarray(weights,dtype=np.int64)
    assert len(counts)==6435 and weights.sum()==8**8
    return counts,weights


def interval(values,weights,tail):
    order=np.argsort(values);cdf=np.cumsum(weights[order])
    return [float(values[order[np.searchsorted(cdf,math.ceil(q*8**8))]]) for q in (tail,1-tail)]


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root,out=Path(args.input),Path(args.output)
    terminal=read(root/'TERMINAL.json')
    assert terminal['passed'] and not terminal['witness'] and len(terminal['cases'])==8
    out.mkdir(parents=True,exist_ok=False);figures=out/'figures';figures.mkdir()
    counts,weights=compositions();cases={};rows=[];contrasts=[];bindings={}
    for entry in terminal['cases']:
        assert [r['actual_terminal_exit'] for r in entry['commands']]==[0,0]
        folder=Path(entry['folder']);data=read(folder/'summary.json');check=read(folder/'independent.json')
        assert not data['witness'] and not check['witness'] and check['passed']
        assert check['summary_sha256']==sha(folder/'summary.json')
        assert len(data['banks'])==8 and all(r['points']==4096 for r in data['banks'])
        identity=(data['case'],data['split']);assert identity not in cases;cases[identity]=data
        bindings[str(folder/'summary.json')]=sha(folder/'summary.json')
        for result in data['pooled']['rows']:
            rows.append(dict(case=data['case'],split=data['split'],qualified=data['all_numerically_qualified'],
                native_fp32_sw2=data['pooled']['native_fp32_sw2'],
                crossed_fp32_sw2=data['pooled']['crossed_fp32_sw2'],
                native_fp64_sw2=data['pooled']['derivative']['sw2'],
                precision_direction=data['pooled']['direction_robust_to_precision'],**result))
        actual=np.array([r['rows'][-1]['actual_loss_change'] for r in data['banks']])
        predicted=np.array([r['rows'][-1]['predicted_loss_change'] for r in data['banks']])
        contrasts.append(dict(case=data['case'],split=data['split'],actual=actual.tolist(),predicted=predicted.tolist(),
            actual_mean=float(actual.mean()),actual_sd=float(actual.std(ddof=1)),
            predicted_mean=float(predicted.mean()),residual_mean=float((predicted-actual).mean()),
            actual_ci95=interval(counts@actual/8,weights,.025),
            actual_ci_adjusted=interval(counts@actual/8,weights,.05/16),
            residual_ci95=interval(counts@(predicted-actual)/8,weights,.025),
            residual_ci_adjusted=interval(counts@(predicted-actual)/8,weights,.05/16),
            bank_squared_fraction=1-float(np.sum((predicted-actual)**2)/np.sum(actual**2)) if np.any(actual) else None))
    primary=[r for r in rows if r['case']=='product_rk4_trained' and r['coefficient']==1.]
    assert len(primary)==2 and len(rows)==32
    explained=all(r['qualified'] and r['precision_direction'] and r['sign_correct'] and
                  r['target_fraction'] is not None and r['target_fraction']>0 for r in primary)
    summary=dict(source_bindings=bindings,all_numerically_qualified=terminal['all_numerically_qualified'],
        primary_full_scale_explained=explained,rows=rows,contrasts=contrasts,
        paired_banks=8,samples_per_bank=4096,pooled_samples=32768,training_seed=0,
        primary_case='product_rk4_trained',source_sha256=sha(Path(__file__)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    order=['ring_midpoint_trained','ring_rk4_trained','product_midpoint_trained','product_rk4_trained']
    names=['Ring8 / midpoint weights','Ring8 / RK4 weights','Product16 / midpoint weights','Product16 / RK4 weights']
    colors=['#0072B2','#D55E00']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for axis,case,name in zip(axes.flat,order,names,strict=True):
        for split,color in zip(['selection','confirmation'],colors,strict=True):
            data=cases[(case,split)]
            x=[0.]+[r['coefficient'] for r in data['pooled']['rows']]
            axis.plot(x,[0.]+[r['actual_loss_change'] for r in data['pooled']['rows']],'-o',color=color,label=split+' actual')
            axis.plot(x,[0.]+[r['predicted_loss_change'] for r in data['pooled']['rows']],'--',color=color,label=split+' tangent')
        axis.axhline(0,color='.6',linewidth=.7);axis.set_title(name)
        axis.set_xlabel('Solver interpolation lambda');axis.set_ylabel('Pooled squared SW2 change')
        axis.ticklabel_format(axis='y',style='sci',scilimits=(0,0));axis.legend(fontsize=8)
    for suffix in ('png','pdf'):fig.savefig(figures/('target-change-calibration.'+suffix),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for axis,case,name in zip(axes.flat,order,names,strict=True):
        limits=[]
        for split,color in zip(['selection','confirmation'],colors,strict=True):
            row=next(r for r in contrasts if r['case']==case and r['split']==split)
            axis.scatter(row['actual'],row['predicted'],color=color,label=split,s=28)
            limits.extend(row['actual']+row['predicted'])
        lower,upper=min(limits),max(limits);pad=max((upper-lower)*.1,1e-8)
        axis.plot([lower-pad,upper+pad],[lower-pad,upper+pad],color='.4',linestyle='--')
        axis.axhline(0,color='.7',linewidth=.7);axis.axvline(0,color='.7',linewidth=.7)
        axis.set_title(name);axis.set_xlabel('Actual bank squared SW2 change')
        axis.set_ylabel('Unfitted tangent prediction at lambda=1')
        axis.ticklabel_format(style='sci',scilimits=(0,0));axis.legend(fontsize=8)
    for suffix in ('png','pdf'):fig.savefig(figures/('paired-bank-predictions.'+suffix),dpi=180)
    plt.close(fig)
    table=['| Case | Split | Native SW2 | Crossed SW2 | Predicted sign | Target fraction | Numerical |',
           '|---|---|---:|---:|---|---:|---|']
    for row in rows:
        if row['coefficient']!=1:continue
        fraction='undefined' if row['target_fraction'] is None else f"{row['target_fraction']:.3f}"
        table.append(f"| {row['case']} | {row['split']} | {row['native_fp32_sw2']:.3f} | {row['crossed_fp32_sw2']:.3f} | {'correct' if row['sign_correct'] else 'wrong'} | {fraction} | {'qualified' if row['qualified'] else 'unresolved'} |")
    (out/'table.md').write_text('\n'.join(table)+'\n')
    decision=('The unfitted learned-field tangent passes the registered primary full-scale explanation on both splits.' if explained else
              'The unfitted learned-field tangent fails the registered primary full-scale explanation on at least one split. Keep the observed solver effect; reject this full-scale linear explanation.')
    (out/'analysis-report.md').write_text('# Does the learned-field tangent explain the solver effect?\n\n'+decision+
        '\n\nAll four frozen20k models, both new splits and every source are retained. No model was trained. '
        'Numerical calibration and target-quality explanation are separate.\n\n'
        '![Target calibration](figures/target-change-calibration.png)\n\n'
        'Solid lines are actual pooled target-functional changes; dashed lines are the derivative prediction with no fitted scale. '
        'A small-lambda agreement cannot establish an explanation at lambda1.\n\n'+ '\n'.join(table)+
        '\n\n![Paired banks](figures/paired-bank-predictions.png)\n\n'
        'Each point is one paired source/target bank. The diagonal denotes an exact full-scale prediction; the axes separate sign failures from magnitude failures. '
        'These banks are conditional evaluation draws, not independent model training runs.\n\n'
        '## Claim candidates\n\n- Claim: The registered primary full-scale transport account '+('passes.' if explained else 'fails.')+
        '\n  - Evidence: Complete bank outputs, independent NumPy reconstruction, numerical gates and both pooled splits.\n'
        '  - Allowed wording: The unfitted first derivative predicts, or fails to predict, these controlled same-weight solver changes.\n'
        '  - Forbidden stronger wording: A general ODE-specific architecture advantage or image-generation improvement.\n'
        '  - Uncertainty: Fixed learned weights, finite target samples and a local expansion.\n'
        '  - Next check: '+('Test transport of independently derived hidden defects, including RK4.' if explained else 'Inspect whether finite solver interpolation needs nonlinear transport before replacing measured field changes by an ODE defect.')+
        '\n  - Decision: '+('keep the transport account for the next causal link.' if explained else 'reject the full-scale linear explanation; preserve the discrete effect.')+'\n')
    (out/'stats-appendix.md').write_text('# Statistics and numerical scope\n\n'
        'One fixed seed0 training trajectory per retained model. Eight paired4096-source evaluation banks per task/split,256directions; pooled32768 endpoints. '
        'Primary quantity is squared SW2 change. Mean/SD, paired95% percentile intervals and nominal eight-contrast-adjusted intervals are in summary.json. '
        'All6435 multinomial count vectors enumerate8^8 ordered resamples exactly. Family tail=.05/16; actual changes and prediction residuals are separate eight-comparison families. '
        'Percentile coverage is empirical/conditional, not guaranteed population coverage. No training-seed inference or architecture superiority test. '
        'Pooled scores and bank-mean differences have different comparison units. Undefined zero-signal fractions remain undefined. '
        'Every numerical failure and measured FP32/FP64 precision effect remains in the full cohort.\n')
    (out/'figure-catalog.md').write_text('# Figure purposes\n\n'
        '- target-change-calibration: all cases/splits, actual and predicted pooled squaredSW2 changes over fixed lambda. '
        'Notice curvature, sign and lambda1 mismatch. No error bars: these are pooled empirical quantities. Determines whether the local account reaches the real intervention.\n'
        '- paired-bank-predictions: every full-scale bank effect against its unfitted prediction. '
        'Notice deviations from identity and signs across the two common splits. Points are conditional banks, never training replicates. Tests whether pooled agreement conceals bank-level failure.\n')
    print(json.dumps(dict(primary_full_scale_explained=explained,rows=len(rows),output=str(out))))


if __name__=='__main__':main()
