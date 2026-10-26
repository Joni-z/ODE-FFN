"""Plot audited saved trajectories without rerunning model evaluation."""
from __future__ import annotations
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


def main(parent:Path,new:Path,output:Path):
    rows=list(csv.DictReader((output/'comparison.csv').open()))
    curves=[r for r in json.loads((parent/'training-curves.json').read_text()) if 'oracle_mse' in r]
    bindings={}
    for dataset in DATASETS:
        for family in ('ode_k2','swiglu128','swiglu192'):
            arm=family+'_directional'
            path=new/'runs'/dataset/arm/'metrics.jsonl'
            bindings[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            for line in path.read_text().splitlines():
                r=json.loads(line)
                if 'oracle_mse' in r:
                    curves.append(dict(dataset=dataset,arm=arm,**r))
    for r in rows:
        found=[c for c in curves if c['dataset']==r['dataset'] and c['arm']==r['arm'] and c['step']==5000]
        assert len(found)==1 and found[0]['oracle_mse']==float(r['oracle_mse'])
    (output/'training-curves.json').write_text(json.dumps(curves,indent=2,allow_nan=False)+'\n')
    colors={'ode_k2':'#D55E00','swiglu128':'#009E73','swiglu192':'#0072B2','swiglu384':'#777777'}
    styles={'direct':':','gradient':'-','directional':'--'}
    fig,axes=plt.subplots(1,3,figsize=(14,4.3))
    for ax,dataset,title in zip(axes,DATASETS,TITLES):
        for r in rows:
            if r['dataset']!=dataset:continue
            cc=sorted([c for c in curves if c['dataset']==dataset and c['arm']==r['arm']],key=lambda c:c['step'])
            ax.plot([c['step'] for c in cc],[c['oracle_mse'] for c in cc],
                color=colors[r['family']],linestyle=styles[r['readout']],label=r['arm'])
        ax.set(title=title,xlabel='Optimizer update',yscale='log')
        ax.grid(alpha=.18)
    axes[0].set_ylabel('Full validation oracle MSE')
    axes[1].legend(ncol=2,fontsize=7,loc='upper right')
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(output/f'learning-curves.{suffix}',dpi=180)
    plt.close(fig)
    spectra=list(csv.DictReader((parent/'p-trajectories.csv').open()))
    spectra+=list(csv.DictReader((output/'new-p-trajectories.csv').open()))
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for ax,dataset,title in zip(axes,DATASETS,TITLES):
        for readout in ('direct','gradient','directional'):
            rr=[r for r in spectra if r['dataset']==dataset and r['arm']=='ode_k2_'+readout]
            steps=sorted({int(r['step']) for r in rr})
            arrays=[np.array([float(r['sigma_max']) for r in rr if int(r['step'])==step]) for step in steps]
            assert len(steps)==4 and all(len(a)==6 for a in arrays)
            ax.plot(steps,[a.mean() for a in arrays],linestyle=styles[readout],label=readout)
            ax.fill_between(steps,[a.min() for a in arrays],[a.max() for a in arrays],alpha=.12)
        ax.set(title=title,xlabel='Optimizer update')
        ax.grid(alpha=.18)
    axes[0].set_ylabel('Largest singular value of P')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(output/f'p-trajectories.{suffix}',dpi=180)
    plt.close(fig)
    (output/'plot-input-bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',endpoint_curves=len(rows),new_curve_files=len(bindings),P_rows=len(spectra))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-results',type=Path,required=True)
    p.add_argument('--new-output',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    a=p.parse_args();main(a.parent_results.resolve(),a.new_output.resolve(),a.results.resolve())
