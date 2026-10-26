"""Plot audited additive-generator endpoints, training and local-map witnesses."""
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
ARMS=('scalar_mul','scalar_add','vector_mul','vector_add','swiglu_budget','swiglu')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def plot(studies,certificate,output):
    figures=output/'figures';figures.mkdir(parents=True,exist_ok=True)
    rows=[];curves={};inputs={}
    for path in sorted(studies.glob('*/audit/audit.json')):
        audit=json.loads(path.read_text());assert audit['state']=='PASS'
        inputs[str(path)]=sha(path);rows.extend(audit['rows'])
        study=path.parent.parent
        for row in audit['rows']:
            config=study/'configs'/f"{row['label']}.json";cfg=json.loads(config.read_text())
            metric=study/'runs'/cfg['experiment_id']/row['dataset']/row['model']/'seed000/metrics.jsonl'
            inputs[str(metric)]=sha(metric)
            curves[row['dataset'],row['label']]=[json.loads(l) for l in metric.read_text().splitlines()]
    assert len(rows)==18
    with (output/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(3,3,figsize=(12,9),layout='constrained')
    for j,(dataset,title) in enumerate(zip(DATASETS,TITLES)):
        data={r['label']:r for r in rows if r['dataset']==dataset}
        for i,(metric,label) in enumerate([('validation_oracle_mse','Oracle MSE'),('jacobian_error','Jacobian error'),('validation_sw2','Validation SW2')]):
            ax=axes[i,j];denominator=min(data[a][metric] for a in ARMS[-2:])
            values=[data[a][metric]/denominator for a in ARMS]
            ax.bar(np.arange(6),values,color=['C0','C1','C0','C1','.6','.3'])
            ax.axhline(1,color='black',ls='--',lw=.8)
            ax.set_xticks(np.arange(6),['S mul','S add','V mul','V add','Swi96','Swi128'],rotation=45,ha='right',fontsize=8)
            ax.set(ylabel=label+' / better SwiGLU',title=title if i==0 else '')
            ax.grid(axis='y',alpha=.15)
    for ext in ('png','pdf'):fig.savefig(figures/f'generator-endpoints.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,dataset,title in zip(axes,DATASETS[1:],TITLES[1:]):
        for k,arm in enumerate(ARMS):
            curve=curves[dataset,arm]
            ax.plot([r['step'] for r in curve],[r['validation_oracle_mse'] for r in curve],
                label=arm,color=f'C{k}' if k<4 else ('.5' if k==4 else '.1'),ls='-' if k<4 else '--')
        ax.set(title=title,xlabel='Update',ylabel='Validation oracle MSE',yscale='log');ax.grid(alpha=.15)
    axes[1].legend(fontsize=8,frameon=False)
    for ext in ('png','pdf'):fig.savefig(figures/f'generator-learning.{ext}',dpi=180)
    plt.close(fig)
    meta=certificate/'certificate.json';example=certificate/'example.npz'
    record=json.loads(meta.read_text());assert record['state']=='PASS'
    inputs[str(meta)]=sha(meta);inputs[str(example)]=sha(example)
    with np.load(example) as data:
        x=data['x'];velocity=data['velocity'];duration=data['duration']
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    axes[0].plot(x,velocity,label='Analytic latent FM velocity')
    limit=float(np.abs(velocity[np.abs(x)<=3]).max())*1.1
    axes[0].axhline(0,color='.4',lw=.7);axes[0].set(xlim=(-3,3),ylim=(-limit,limit),xlabel='Latent x',ylabel='Velocity',title='Restricted scalar example at t=.5')
    selected=min(record['axes'],key=lambda row:abs(row['mu']-1))
    static=record['center']*selected['fixed_euler_P']
    axes[1].plot(x,static+duration-record['center'],label='Total: cP + (s-c)')
    axes[1].plot(x,duration-record['center'],label='Controller term s-c',ls='--')
    axes[1].axhline(static,label='Fixed cP',color='C2',ls=':')
    axes[1].axhline(0,color='.4',lw=.7)
    axes[1].set(xlim=(-3,3),xlabel='Latent x',ylabel='Euler increment / x',title='Ideal witness; not a trained network')
    for ax in axes:ax.legend(fontsize=8,frameon=False);ax.grid(alpha=.15)
    for ext in ('png','pdf'):fig.savefig(figures/f'local-witness.{ext}',dpi=180)
    plt.close(fig)
    (output/'plot-provenance.json').write_text(json.dumps(dict(script_sha256=sha(Path(__file__)),input_sha256=inputs,
        scope='Audited descriptive endpoints/trajectories. Local witness is a restricted analytic capacity example, not a trained-model prediction or full-network lower bound.'),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('studies','certificate','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();plot(args.studies,args.certificate,args.output)
