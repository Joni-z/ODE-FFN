"""Plot the completed shared/cloned readout factor without model calls."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(results):
    assert json.loads((results/'status.json').read_text())['state']=='COMPLETE'
    rows=list(csv.DictReader((results/'factor.csv').open()))
    curves=json.loads((results/'training-curves.json').read_text())
    for r in rows:
        final=[c for c in curves if c['arm']==r['arm'] and c['step']==5000]
        assert len(final)==1 and final[0]['oracle_mse']==float(r['oracle_mse'])
    families=('ode_k2','swiglu128','swiglu192')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,family in zip(axes,families):
        positions=np.arange(2)
        for j,readout in enumerate(('direct','gradient')):
            values=[float(next(r['oracle_mse'] for r in rows if (r['family'],r['layout'],r['readout'])==(family,layout,readout))) for layout in ('cloned','shared')]
            ax.bar(positions+(-.18 if j==0 else .18),values,width=.36,label=readout)
        ax.set(title=family,xticks=positions,xticklabels=['Cloned independent','Shared'],yscale='log')
        ax.tick_params(axis='x',labelsize=8)
    axes[0].set_ylabel('Full validation oracle MSE');axes[0].legend(fontsize=8)
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(results/f'factor-comparison.{suffix}',dpi=180)
    plt.close(fig)
    colors={'shared':'#D55E00','cloned':'#0072B2'}
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,family in zip(axes,families):
        for layout in ('shared','cloned'):
            for readout in ('direct','gradient'):
                arm=family+'_'+layout+'_'+readout
                rr=[r for r in curves if r['arm']==arm and 'oracle_mse' in r]
                ax.plot([r['step'] for r in rr],[r['oracle_mse'] for r in rr],color=colors[layout],
                    linestyle='-' if readout=='gradient' else ':',label=layout+'/'+readout)
        ax.set(title=family,xlabel='Optimizer update',yscale='log');ax.grid(alpha=.18)
    axes[0].set_ylabel('Full validation oracle MSE');axes[0].legend(fontsize=7)
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(results/f'learning-curves.{suffix}',dpi=180)
    plt.close(fig)
    spectra=list(csv.DictReader((results/'p-trajectories.csv').open()))
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    for ax,readout in zip(axes,('direct','gradient')):
        for layout in ('shared','cloned'):
            arm='ode_k2_'+layout+'_'+readout
            steps=sorted({int(r['step']) for r in spectra if r['arm']==arm})
            values=[np.array([float(r['sigma_max']) for r in spectra if r['arm']==arm and int(r['step'])==step]) for step in steps]
            assert all(len(v)==(2 if layout=='shared' else 6) for v in values)
            ax.plot(steps,[v.mean() for v in values],label=layout,color=colors[layout])
            ax.fill_between(steps,[v.min() for v in values],[v.max() for v in values],color=colors[layout],alpha=.15)
        ax.set(title=readout,xlabel='Optimizer update');ax.grid(alpha=.18)
    axes[0].set_ylabel('Largest singular value of P');axes[0].legend(fontsize=8)
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(results/f'p-trajectories.{suffix}',dpi=180)
    plt.close(fig)
    print(json.dumps(dict(status='PASS',endpoints_verified=len(rows),figures=3)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);a=p.parse_args();main(a.results.resolve())
