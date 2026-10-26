"""Plot complete follow-up evidence without selecting flattering checkpoints."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS=['scalar_k1','vector_k1','swiglu']
DISPLAY=['Scalar K1','Vector K1','SwiGLU']
COLORS=['#0072B2','#D55E00','#009E73']
STUDIES=['long_ring_20261024','rotated16_20261024']
TITLES=['Ring8 (2D)','Rotated Ring8 + Gaussian (16D)']


def plot(root: Path,rotation: np.ndarray,output: Path) -> None:
    output.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        for suffix in ('png','pdf'):fig.savefig(output/f'{name}.{suffix}',dpi=200,bbox_inches='tight')
        plt.close(fig)
    summaries=[json.loads((root/s/'heldout/summary.json').read_text()) for s in STUDIES]
    fig,axes=plt.subplots(2,3,figsize=(11,6),constrained_layout=True)
    for i,summary in enumerate(summaries):
        rows={r['label']:r for r in summary['rows']}
        for j,(metric,title) in enumerate([('test_field_mse','Test velocity MSE'),
                                           ('test_jacobian_error','Test Jacobian squared error'),
                                           ('sw2','Test sample SW2, Heun32')]):
            values=[rows[k]['sampling'][0]['sw2'] if metric=='sw2' else rows[k][metric] for k in LABELS]
            ax=axes[i,j]
            ax.bar(np.arange(3),values,color=COLORS,width=.65)
            ax.set_xticks(np.arange(3),DISPLAY,rotation=15)
            ax.set_title(TITLES[i]+'\n'+title,fontsize=10)
            ax.set_ylim(0,max(values)*1.25)
            for index,value in enumerate(values):ax.text(index,value,f'{value:.4g}',ha='center',va='bottom',fontsize=9)
            if metric=='sw2':
                ax.axhline(rows[LABELS[0]]['reference_sw2_floor'],color='0.3',ls=':',label='Target vs target')
                ax.legend(fontsize=8)
            ax.spines[['right','top']].set_visible(False)
    save(fig,'heldout-comparison')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for ax,study,title in zip(axes,STUDIES,TITLES):
        provenance=json.loads((root/study/'provenance.json').read_text())
        dataset=provenance['plan']['datasets'][0]
        for label,name,color in zip(LABELS,DISPLAY,COLORS):
            cfg=json.loads((root/study/'configs'/f'{label}.json').read_text())
            path=root/study/'runs'/cfg['experiment_id']/dataset/cfg['models'][0]/'seed000/metrics.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            ax.semilogy([r['step'] for r in rows],[r['validation_oracle_mse'] for r in rows],color=color,label=name)
        ax.set(title=title,xlabel='Training updates',ylabel='Validation oracle MSE')
        ax.grid(alpha=.2);ax.legend();ax.spines[['right','top']].set_visible(False)
    save(fig,'training-curves')
    fig,axes=plt.subplots(2,4,figsize=(11,5.5),constrained_layout=True)
    for i,(study,summary,title) in enumerate(zip(STUDIES,summaries,TITLES)):
        dataset=summary['rows'][0]['dataset']
        with np.load(root/study/'heldout'/f'{dataset}-inputs.npz') as bank:target=bank['target']
        arrays=[target]
        for label in LABELS:
            with np.load(root/study/'heldout'/dataset/label/'samples-heun32.npz') as a:arrays.append(a['samples'])
        for j,(ax,points,name) in enumerate(zip(axes[i],arrays,['Target',*DISPLAY])):
            if i:points=points@rotation
            ax.scatter(points[:,0],points[:,1],s=1.5,alpha=.25,c='0.25' if j==0 else COLORS[j-1],rasterized=True)
            ax.set_title(title+'\n'+name,fontsize=9);ax.set_aspect('equal')
            ax.set(xlim=(-4,4),ylim=(-4,4),xlabel='Mode coordinate 1',ylabel='Mode coordinate 2')
    save(fig,'mode-samples')
    # Rate/P trajectories are retained rather than reducing the whole study to endpoint quality.
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for ax,study,title in zip(axes,STUDIES,TITLES):
        cfg=json.loads((root/study/'configs/vector_k1.json').read_text())
        dataset=json.loads((root/study/'provenance.json').read_text())['plan']['datasets'][0]
        p=root/study/'runs'/cfg['experiment_id']/dataset/'ode_vector/seed000/metrics.jsonl'
        rows=[json.loads(line) for line in p.read_text().splitlines()]
        for index in range(6):
            ax.plot([r['step'] for r in rows],[r['ode_layers'][index]['effective']['sigma_max'] for r in rows],
                    label=f'Block {index//2+1}, P{index%2+1}')
        ax.set(title=title+' (vector K1)',xlabel='Training updates',ylabel='Largest singular value of P')
        ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2);ax.spines[['right','top']].set_visible(False)
    save(fig,'operator-trajectories')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--rotated-data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads((args.rotated_data/'manifest.json').read_text())
    rotation=np.asarray(manifest['datasets']['rotated_gmm_ring8_16d']['distribution']['rotation'])
    plot(args.input,rotation,args.output)
