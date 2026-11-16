"""Test prebound field-based solver rankings on all common sampling banks."""
import argparse
import json
import math
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_rk4 import compositions
from verify_rk4_samples import sha


def read(path: Path):
    return json.loads(path.read_text())


def dump(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def interval(values: np.ndarray, weights: np.ndarray, q: float) -> list:
    order = np.argsort(values)
    cdf = np.cumsum(weights[order])
    return [float(values[order[np.searchsorted(cdf,math.ceil(p*8**8))]]) for p in [q,1-q]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    root,out = Path(args.input),Path(args.output)
    terminal = read(root/'ops/TERMINAL.json')
    native_terminal = [r for r in terminal['completed'] if r['phase']=='native']
    assert len(native_terminal)==8 and all(r['actual_terminal_exit']==0 for r in native_terminal)
    validation = read(root/'verification-v2/PASS.json')
    assert validation['passed'] and validation['cases']==8
    assert validation['actual_child_exits']==[0]*16 and validation['original_samples_and_outputs_unchanged']
    parent = root.parent
    controls = read(parent/'metrics-verified/PASS.json')
    assert controls['passed'] and controls['models']==8
    plan = read(Path(__file__).resolve().parents[3]/'docs/research-assessments/2026-11-16-ode-mechanism/cross-solver-cases.json')
    counts,weights = compositions()
    rows,contrasts = [],[]
    for case in plan['cases']:
        for split in ['selection','confirmation']:
            folder = root/'formal'/split/case['id']
            verified = root/'verification-v2'/split/case['id']/'verified/PASS.json'
            proof = read(verified)
            assert proof['passed'] and not proof['witness']
            assert proof['input_complete_sha256']==sha(folder/'COMPLETE.json')
            assert proof['id']==case['source_id'] and proof['inference_solver']==case['inference_solver']
            native = next(r for r in controls['rows'] if r['id']==case['source_id'] and r['split']==split)
            assert native['checkpoint_sha256']==proof['checkpoint_sha256'] and native['asset_sha256']==proof['asset_sha256']
            delta = np.asarray(proof['metrics']['bank_sw2'])-native['bank_sw2']
            assert delta.shape==(8,)
            bootstrap = counts@delta/8
            q4,q8 = interval(bootstrap,weights,.05/8),interval(bootstrap,weights,.05/16)
            sign = -1 if case['predicted_crossed_sw2_change']=='improve' else 1
            pooled = proof['metrics']['pooled_sw2']-native['pooled_sw2']
            support = bool(sign*delta.mean()>0 and sign*pooled>0)
            resolved = bool(support and (q4[1]<0 if sign<0 else q4[0]>0))
            rows.append(dict(**case,split=split,native=native,crossed=proof['metrics'],
                             verification_sha256=sha(verified),checkpoint_sha256=proof['checkpoint_sha256']))
            contrasts.append(dict(id=case['id'],dataset=case['dataset'],trained_solver=case['trained_solver'],
                inference_solver=case['inference_solver'],split=split,predicted=case['predicted_crossed_sw2_change'],
                bank_differences=delta.tolist(),mean_difference=float(delta.mean()),sd_difference=float(delta.std(ddof=1)),
                pooled_difference=float(pooled),better_banks=int((delta<0).sum()),marginal95=interval(bootstrap,weights,.025),
                four_comparison_adjusted=q4,eight_comparison_sensitivity=q8,
                forecast_point_support=support,forecast_adjusted_support=resolved))
    out.mkdir(parents=True,exist_ok=False)
    (out/'figures').mkdir()
    result = dict(passed=True,rows=rows,contrasts=contrasts,ordinary_controls=[r for r in controls['rows'] if r['family'] in ['free_isru','swiglu64','swiglu192']],
        point_supported=sum(r['forecast_point_support'] for r in contrasts),adjusted_supported=sum(r['forecast_adjusted_support'] for r in contrasts),
        total_forecasts=8,ordered_bootstrap_resamples=8**8,exact_multisets=6435,new_training_runs=0)
    dump(out/'summary.json',result)
    table = '| Task | Trained solver | Inference solver | Field MSE | Pooled SW2 |\n|---|---|---|---:|---:|\n'
    for task,title in [('gmm_ring8_2d','Ring8'),('rotated_product_bimodal_16d','Product16')]:
        for row in [r for r in rows if r['dataset']==task and r['split']=='confirmation']:
            for inference,metrics in [(row['trained_solver'],row['native']),
                                      (row['inference_solver'],row['crossed'])]:
                table += f"| {title} | {row['trained_solver']} | {inference} | {metrics['field_mse']:.3f} | {metrics['pooled_sw2']:.3f} |\n"
        for row in [r for r in result['ordinary_controls'] if r['dataset']==task and r['split']=='confirmation']:
            table += f"| {title} | {row['family']} | native | {row['field_mse']:.3f} | {row['pooled_sw2']:.3f} |\n"
    (out/'table.md').write_text(table)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes = plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
    for column,(task,title) in enumerate([('gmm_ring8_2d','Ring8'),('rotated_product_bimodal_16d','Product16')]):
        group = [r for r in rows if r['dataset']==task and r['split']=='confirmation']
        matrix = np.empty((2,2))
        for i,row in enumerate(group):
            own = 0 if row['trained_solver']=='midpoint' else 1
            matrix[i,own] = row['native']['pooled_sw2']
            matrix[i,1-own] = row['crossed']['pooled_sw2']
        ax = axes[0,column]
        im = ax.imshow(matrix,cmap='YlOrRd',aspect='auto')
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white' if matrix[i,j]>(matrix.min()+matrix.max())/2 else 'black')
        ax.set(xticks=[0,1],xticklabels=['Midpoint4','RK4 × 2'],yticks=[0,1],yticklabels=['Midpoint-trained','RK4-trained'],title=title+' / pooled SW2 ↓')
        fig.colorbar(im,ax=ax,shrink=.8)
        ordinary = [r for r in result['ordinary_controls'] if r['dataset']==task and r['split']=='confirmation']
        values = list(matrix.ravel())+[r['pooled_sw2'] for r in ordinary]
        ax = axes[1,column]
        ax.bar(range(6),values,color=['#2166ac','#80b1d3','#d95f02','#fdb462','#7570b3','#444444'])
        ax.set(xticks=range(6),xticklabels=['M→M','M→R','R→M','R→R','Free ISRU','SwiGLU'],ylabel='Pooled target SW2 ↓',title=title+' / ordinary controls retained')
        ax.tick_params(axis='x',rotation=20)
    for ext in ['png','pdf']:
        fig.savefig(out/'figures'/('crossed-sampling-and-controls.'+ext),dpi=180)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    for ax,task,title in zip(axes,['gmm_ring8_2d','rotated_product_bimodal_16d'],['Ring8','Product16']):
        group = [r for r in contrasts if r['dataset']==task]
        for i,row in enumerate(group):
            lo,hi = row['four_comparison_adjusted']
            color = '#1b9e77' if row['forecast_point_support'] else '#d95f02'
            ax.hlines(i,lo*1000,hi*1000,color=color,lw=2)
            ax.scatter(row['mean_difference']*1000,i,color=color,s=35,zorder=3)
        ax.axvline(0,color='.4',lw=1)
        ax.set(yticks=range(4),yticklabels=[r['trained_solver']+' / '+r['split'] for r in group],
               xlabel='Crossed minus native bank SW2 × 1000',title=title+' / paired differences')
    for ext in ['png','pdf']:
        fig.savefig(out/'figures'/('paired-sampling-forecasts.'+ext),dpi=180)
    plt.close(fig)
    shutil.copyfile(root/'ops/TERMINAL.json',out/'sampling-terminal-original.json')
    shutil.copyfile(root/'verification-v2/PASS.json',out/'verification-terminal.json')
    (out/'analysis-report.md').write_text(f'''# Do field-based solver rankings predict generation?

The prebound field-based ranking is supported in{result['point_supported']}/8model/split
comparisons by both pooledSW2and mean paired-bank direction; adjusted intervals
support{result['adjusted_supported']}/8. Every case, split and bank is retained.
These are frozen-weight inference interventions; no model was retrained.

## Controlled substitution

Each of four20kweight sets is evaluated with the other hidden discrete solver.
Both methods use eight Papplications per local solve. The original native
Heun32final samples replay exactly before substitution; then the copied native
evaluator uses unchanged Heun128, eight4096-source banks and256directions.
Independent NumPy reconstructs every field, sample and mode/gap metric.

![Sampling and controls](figures/crossed-sampling-and-controls.png)

Rows fix trained weights, columns select the inference solver. Bottom labels
mean training→inference(M=midpoint,R=RK4). Ordinary SwiGLUand FreeISRUcontrols
remain unchanged; Ring8SwiGLU64uses Muon, Product16SwiGLU192uses AdamW.
The field diagnostic and sampling test answer different questions: a field-risk
benefit is not counted as a generation gain without the target comparison.

'''+table+'''
![All paired forecasts](figures/paired-sampling-forecasts.png)

Intervals concern paired-bank mean differences; pooled scores are reported
separately. Negative favors the crossed solver. Green means the pooled/mean
point directions follow the prebound prediction; orange means at least one
does not. All original common sources and checkpoint identities are preserved.

## Claim candidates

- Claim: the specific recorded field-based ranking transfers to generation
  in the listed cases and fails in the others.
  - Evidence: complete selection/confirmation banks and independent reconstruction.
  - Allowed wording: retain the per-case dispositions in summary.json.
  - Forbidden stronger wording: universal field-MSEor ODEsolver advantage.
  - Uncertainty: fixed learned weights and conditional sampling-bank variation.
  - Next check: choose the next mechanism from the complete result, not one bank.
  - Decision: keep case-specific evidence and revise any broader claim.
''')
    (out/'stats-appendix.md').write_text('''# Comparison units and exact paired statistics

Four retained20kweight sets, training seed0. Eight new full evaluations:
four crossed solvers by selection/confirmation. Each uses eight common4096-source
banks,256directions,Heun128and32768pooled samples. Native original-solver and
ordinary control metrics are reused only with matching asset/checkpoint hashes.
All eight new metric reconstructions have actual terminal0, as do all evaluations.

Paired bank difference is crossed minus nativeSW2, keeping source/target and
directions identical. Enumerate6435multisets with exact multinomial weights for
all8^8ordered bank resamples. Report marginal95%, four-comparison-adjusted
intervals per split, and8-comparison sensitivity over both splits. These describe
conditional evaluation-bank variation of the retained weights. PooledSW2and
the mean of bankSW2are distinct statistics; both directions must satisfy the
prebound prediction. No arbitrary performance margin or best-arm selection.
''')
    (out/'figure-catalog.md').write_text('''# Figure interpretation

## crossed-sampling-and-controls
Purpose: distinguish fixed weights from inference solver and retain strong
ordinary baselines. Confirmation pooledSW2has no error bars. All heatmap/bar
values come from independently verified raw samples; three decimals are display
precision only. Compare complete rows rather than selecting a single cell.

## paired-sampling-forecasts
Purpose: test the prospective field-based directions on both reserved splits.
Points are mean paired-bankSW2differences; intervals are exact conditional
four-comparison-adjusted bootstrap intervals. Green follows both pooled/mean
point forecasts, orange does not. This tests a concrete mechanism bridge;
field improvement alone is not a sample-quality result.
''')
    print('crossed-sampling analysis complete',result['point_supported'],result['adjusted_supported'],flush=True)


if __name__=='__main__':
    main()
