"""Summarize archived projection moments without treating layers as replications."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


GROUPS = [('vector_k1', 'P', 'ODE P'), ('vector_reparam', 'P', 'Paired P'),
          ('vector_k1', 'clock_out', 'ODE clock out'),
          ('swiglu_budget', 'swiglu_encoder', 'Small Swi encoder'),
          ('swiglu_budget', 'swiglu_decoder', 'Small Swi decoder'),
          ('swiglu', 'swiglu_encoder', 'Full Swi encoder'),
          ('swiglu', 'swiglu_decoder', 'Full Swi decoder')]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(inputs, output):
    output.mkdir(parents=True, exist_ok=True)
    (output/'figures').mkdir(exist_ok=True)
    records, summaries, hashes = [], {}, {}
    for task, path in inputs.items():
        summary = json.loads((path/'summary.json').read_text())
        assert summary['state'] == 'COMPLETE'
        for name, expected in summary['files_sha256'].items():
            assert digest(path/name) == expected
        summaries[task] = summary
        hashes[str(path/'summary.json')] = digest(path/'summary.json')
        for label in summary['labels']:
            for step in summary['steps']:
                with np.load(path/f'{label}-{step}.npz', allow_pickle=False) as archive:
                    responses = [r for r in summary['responses'] if r['label']==label and r['step']==step]
                    for response in responses:
                        name = response['node']
                        matrix = archive[name+'__G__uniform']
                        eigen, basis = np.linalg.eigh((matrix+matrix.T)/2)
                        active = np.flatnonzero(eigen > max(1e-12, eigen[-1]*1e-10))
                        count = max(1, len(active)//4)
                        moment = next(r for r in summary['moments'] if r['label']==label and
                                      r['step']==step and r['panel']=='uniform' and r['node']==name)
                        for loss in ('pair', 'population'):
                            gradient = archive[name+'__gradient__'+loss]
                            projected = gradient @ basis[:, active]
                            standardized = (projected**2).sum(axis=0)/eigen[active]
                            total = float(standardized.sum())
                            records.append(dict(task=task,label=label,step=step,node=name,
                                role=response['role'],loss=loss,rank=len(active),
                                condition=moment['positive_condition'],participation=moment['participation'],
                                raw_low_share=response['losses'][loss]['coherent_gradient_low_input_share'],
                                standardized_low_share=float(standardized[:count].sum()/total) if total else None,
                                standardized_gradient_energy=total))
    with (output/'projection-summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    table=['| Task | Projection group | Layers | Median positive condition | Median raw low share | Median standardized low share |',
           '|---|---|---:|---:|---:|---:|']
    figure, axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for column, task in enumerate(inputs):
        for index,(label,role,title) in enumerate(GROUPS):
            rows=[r for r in records if r['task']==task and r['step']==5000 and r['label']==label and
                  r['role']==role and r['loss']=='population']
            condition=np.array([r['condition'] for r in rows])
            low=np.array([r['standardized_low_share'] for r in rows])
            raw=np.array([r['raw_low_share'] for r in rows])
            for ax, values in ((axes[0,column],condition),(axes[1,column],low*100)):
                center=np.median(values)
                ax.errorbar(index,center,yerr=[[center-values.min()],[values.max()-center]],
                            fmt='o',capsize=3,color=f'C{index}',markersize=5)
            table.append(f'| {task} | {title} | {len(rows)} | {np.median(condition):.0f} | {np.median(raw)*100:.4f}% | {np.median(low)*100:.2f}% |')
        axes[0,column].set_yscale('log')
        axes[0,column].set_title(task+' — step 5000')
        axes[0,column].set_ylabel('Positive-spectrum condition')
        axes[1,column].set_ylabel('Standardized gradient energy in lowest quarter (%)')
        axes[1,column].axhline(25,color='.6',ls=':',lw=1)
        for ax in axes[:,column]:
            ax.set_xticks(range(len(GROUPS)),[g[2] for g in GROUPS],rotation=35,ha='right')
            ax.grid(axis='y',alpha=.2)
    figure.savefig(output/'figures/conditioning-and-residual-alignment.png',dpi=180)
    figure.savefig(output/'figures/conditioning-and-residual-alignment.pdf')
    plt.close(figure)
    figure, axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,task in zip(axes,inputs):
        for index,(label,role,title) in enumerate([GROUPS[0],GROUPS[1],GROUPS[3]]):
            medians,minimum,maximum=[],[],[]
            for step in (0,1000,5000):
                values=[r['participation'] for r in records if r['task']==task and r['step']==step and
                        r['label']==label and r['role']==role and r['loss']=='population']
                medians.append(np.median(values));minimum.append(min(values));maximum.append(max(values))
            ax.plot([0,1000,5000],medians,'o-',label=title,color=f'C{index}')
            ax.fill_between([0,1000,5000],minimum,maximum,alpha=.12,color=f'C{index}')
        ax.set(title=task,xlabel='Training update',ylabel='Input-moment participation')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    figure.savefig(output/'figures/participation-trajectory.png',dpi=180)
    figure.savefig(output/'figures/participation-trajectory.pdf')
    plt.close(figure)
    (output/'exact-table.md').write_text('\n'.join(table)+'\n')
    (output/'provenance.json').write_text(json.dumps(dict(input_summaries_sha256=hashes,
        plot_source_sha256=digest(Path(__file__)),
        note='Standardized-gradient readout is a post-hoc scale correction, not a preregistered training outcome. Layer ranges are descriptive, not confidence intervals.',
        outputs_sha256={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file() and p.name!='provenance.json'}),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ring',type=Path,required=True)
    parser.add_argument('--product16',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summarize(dict(Ring8=args.ring,Product16=args.product16),args.output)
