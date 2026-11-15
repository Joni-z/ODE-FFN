"""Render saved spectral rows locally; no plotting dependency on training hosts."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
a=p.parse_args();source=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
rows=json.loads((source/'layers.json').read_text())
selected=[r for r in rows if r['family'] in ('pd_isru_k1','free_isru')]
labels={'pd_isru_k1':'ISRU-D','free_isru':'Free ISRU'}
colors={'pd_isru_k1':'#0072B2','free_isru':'#D55E00'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
for dataset in sorted({r['dataset'] for r in selected}):
    fig,axes=plt.subplots(2,3,figsize=(11.5,6),layout='constrained')
    for ax,(kind,metric) in zip(axes.flat,[(k,m) for k in ('weight','delta') for m in ('stable_rank','entropy_rank','condition_number')]):
        for family in labels:
            for optimizer,ls in [('adamw','-'),('muon','--')]:
                sub=[r for r in selected if r['dataset']==dataset and r['family']==family and r['optimizer']==optimizer and r['kind']==kind]
                steps=sorted({r['step'] for r in sub});values=[]
                for step in steps:
                    v=[r[metric] for r in sub if r['step']==step and r[metric] is not None]
                    values.append(np.mean(v) if v else np.nan)
                ax.plot(steps,values,color=colors[family],linestyle=ls,label=labels[family]+'/'+optimizer)
        if metric=='condition_number':ax.set_yscale('log')
        ax.set(title=('Learned W' if kind=='weight' else 'Actual update')+': '+metric.replace('_',' '),xlabel='Training update')
    axes[0,0].legend(fontsize=7);fig.savefig(out/'product16-rank-trajectories.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    for ax,family in zip(axes,labels):
        for opt in ('adamw','muon'):
            spectra=np.array([r['singular_values'] for r in selected if r['dataset']==dataset and r['family']==family and r['optimizer']==opt and r['kind']=='weight' and r['step']==20000])
            normalized=spectra/spectra[:,0,None]
            ax.semilogy(np.arange(1,65),np.mean(normalized,axis=0),label=opt)
        ax.set(title=labels[family]+' at20k',xlabel='Singular value index',ylabel='Mean sigma_i / sigma_1');ax.legend()
    fig.savefig(out/'product16-final-weight-spectrum.png',dpi=180);plt.close(fig)
for name in ('endpoints.json','binding.json','layers.csv'):
    (out/name).write_bytes((source/name).read_bytes())
print(json.dumps({'rows':len(rows),'figures':2,'output':str(out)}))
