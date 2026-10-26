"""Plot audited identity-location outcomes and representation-dependent spectra."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


DATASETS=('gaussian_aniso_2d','gmm_ring8_2d','rotated_product_bimodal_16d')
TITLES=('Gaussian2D','Ring8','Product16')
PATTERNS=('mm','mi','im','ii')


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def plot(studies,output):
    output.mkdir(parents=True,exist_ok=True);figures=output/'figures';figures.mkdir(exist_ok=True)
    rows=[];spectra=[];inputs=[];curves={}
    for study in sorted(studies.iterdir()):
        audit_path=study/'audit/audit.json'
        if not audit_path.exists():continue
        audit=json.loads(audit_path.read_text());assert audit['state']=='PASS'
        centered=study/'centered-operators.json';measure=json.loads(centered.read_text())
        assert measure['state']=='COMPLETE' and measure['native_audit_sha256']==digest(audit_path)
        inputs.extend([audit_path,centered]);spectra.extend(measure['rows'])
        for row in audit['rows']:
            rows.append(row);cfg_path=study/'configs'/f"{row['label']}.json";cfg=json.loads(cfg_path.read_text())
            path=study/'runs'/cfg['experiment_id']/row['dataset']/row['model']/'seed000/metrics.jsonl'
            inputs.extend([cfg_path,path]);curves[(row['dataset'],row['label'])]=[json.loads(l) for l in path.read_text().splitlines()]
    assert len(rows)==30
    with (output/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    ratios=[]
    for family in ('scalar','vector'):
        for dataset in DATASETS:
            selected={r['label']:r for r in rows if r['dataset']==dataset}
            denominator=min(selected[a]['validation_oracle_mse'] for a in ('swiglu_budget','swiglu'))
            ratios.extend(selected[family+'_'+p]['validation_oracle_mse']/denominator for p in PATTERNS)
    norm=matplotlib.colors.Normalize(vmin=0,vmax=max(ratios))
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for i,family in enumerate(('scalar','vector')):
        for j,(dataset,title) in enumerate(zip(DATASETS,TITLES)):
            selected={r['label']:r for r in rows if r['dataset']==dataset}
            denominator=min(selected[a]['validation_oracle_mse'] for a in ('swiglu_budget','swiglu'))
            values=np.array([selected[family+'_'+p]['validation_oracle_mse'] for p in PATTERNS]).reshape(2,2)
            ax=axes[i,j];im=ax.imshow(values/denominator,norm=norm,cmap='YlOrRd')
            ax.set_xticks([0,1],['Map','Increment']);ax.set_yticks([0,1],['Map','Increment'])
            ax.set(xlabel='Second solve',ylabel='First solve',title=family.title()+' / '+title)
            for a in range(2):
                for b in range(2):
                    ax.text(b,a,f'{values[a,b]:.5f}\n{values[a,b]/denominator:.2f}x Swi',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=axes,label='Oracle MSE / better SwiGLU',shrink=.8)
    for suffix in ('png','pdf'):fig.savefig(figures/f'location-factorial.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,6.5),layout='constrained')
    for i,family in enumerate(('scalar','vector')):
        for j,(dataset,title) in enumerate(zip(DATASETS[1:],TITLES[1:])):
            ax=axes[i,j]
            for k,p in enumerate(PATTERNS):
                curve=curves[(dataset,family+'_'+p)]
                ax.plot([v['step'] for v in curve],[v['validation_oracle_mse'] for v in curve],label=p.upper(),color=f'C{k}')
            for label,color,style in [('swiglu_budget','.35','--'),('swiglu','.05',':')]:
                curve=curves[(dataset,label)]
                ax.plot([v['step'] for v in curve],[v['validation_oracle_mse'] for v in curve],label=label,color=color,ls=style)
            ax.set(title=family.title()+' / '+title,xlabel='Update',ylabel='Oracle MSE',yscale='log');ax.grid(alpha=.15)
    axes[0,1].legend(fontsize=8,frameon=False)
    for suffix in ('png','pdf'):fig.savefig(figures/f'location-learning.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for label,title,style in [('vector_mm','MM','-'),('vector_ii','II','--')]:
        row=next(r for r in spectra if r['dataset']==DATASETS[2] and r['label']==label and r['step']==0 and r['layer']=='blocks.0.ffn.ode1')
        for ax,kind in zip(axes,('raw','centered')):
            ax.plot(row[kind]['singular_values'],ls=style,label=title)
    for ax,title in zip(axes,('Raw P','Reference map gamma*I + cP')):
        ax.set(title=title,xlabel='Ordered singular index',ylabel='Singular value');ax.legend();ax.grid(alpha=.2)
    for suffix in ('png','pdf'):fig.savefig(figures/f'initial-spectral-equivalence.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    summaries=[]
    for j,(dataset,title) in enumerate(zip(DATASETS[1:],TITLES[1:])):
        for k,p in enumerate(PATTERNS):
            selected=[r for r in spectra if r['dataset']==dataset and r['label']=='vector_'+p]
            steps=sorted({r['step'] for r in selected})
            for i,kind in enumerate(('raw','centered')):
                means=[np.mean([r[kind]['sigma_max'] for r in selected if r['step']==step]) for step in steps]
                axes[i,j].plot(steps,means,label=p.upper(),color=f'C{k}')
                axes[i,j].set(title=title+' / '+('P' if kind=='raw' else 'reference map'),xlabel='Update',ylabel='Mean spectral norm')
                axes[i,j].grid(alpha=.2)
    axes[0,1].legend(fontsize=8,frameon=False)
    for suffix in ('png','pdf'):fig.savefig(figures/f'centered-map-trajectories.{suffix}',dpi=180)
    plt.close(fig)
    for dataset in DATASETS:
        for family in ('scalar','vector'):
            for p in PATTERNS:
                for step in (0,1000,2000,3000,4000,5000):
                    r=[r for r in spectra if r['dataset']==dataset and r['label']==family+'_'+p and r['step']==step]
                    summaries.append(dict(dataset=dataset,label=family+'_'+p,step=step,
                        raw_sigma=np.mean([v['raw']['sigma_max'] for v in r]),
                        centered_sigma=np.mean([v['centered']['sigma_max'] for v in r]),
                        raw_cv=np.mean([v['raw']['singular_cv'] for v in r]),
                        centered_cv=np.mean([v['centered']['singular_cv'] for v in r])))
    with (output/'operator-summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    (output/'plot-provenance.json').write_text(json.dumps(dict(source_sha256=digest(Path(__file__)),
        input_sha256={str(p):digest(p) for p in sorted(set(inputs))},
        scope='Descriptive factorial endpoints, trajectories and constant-reference-duration matrices; no statistical inference or full-Jacobian stability claim.'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--studies',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();plot(args.studies,args.output)
