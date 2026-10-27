"""Visualize the fixed-screen endpoints, learned matrices and full curves."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True);args=parser.parse_args();root=args.results
out=root/'figures';out.mkdir(exist_ok=False)
read=lambda name:list(csv.DictReader((root/name).open()))
rows=read('comparison.csv');laws=['gaussian_aniso_2d','gmm_ring8_2d','rotated_product_bimodal_16d','sign_code','half_mixture']
labels=['Gaussian2D','Ring8','Product16','Dependent signs','Half mixture']
arms=['transport','diagonal','swiglu128','swiglu143','swiglu160'];names=['Exact transport','Diagonal scaling','SwiGLU128 (reused)','SwiGLU143','SwiGLU160 (reused)']
colors=['#c44e52','#999999','#55a868','#4c72b0','#8172b2']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(13,4.2))
for ax,key,title in zip(axes,['oracle_mse','jacobian_error','sw2'],['Oracle field MSE','Complete velocity Jacobian error','Generated-sample SW2']):
    for index,(arm,name,color) in enumerate(zip(arms,names,colors)):
        values=[]
        for law in laws:
            group=[r for r in rows if r['law']==law];denominator=float(next(r[key] for r in group if r['arm']=='swiglu143'))
            values.append(float(next(r[key] for r in group if r['arm']==arm))/denominator)
        ax.bar(np.arange(5)+(index-2)*.15,values,.14,label=name,color=color)
    ax.axhline(1,color='black',linewidth=.8,linestyle='--');ax.set_title(title)
    ax.set_xticks(range(5),labels,rotation=30,ha='right');ax.set_ylabel('Ratio to matched-budget SwiGLU143');ax.grid(axis='y',alpha=.2)
handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,ncol=5,loc='upper center',bbox_to_anchor=(.5,1.02),frameon=False)
fig.tight_layout(rect=(0,0,1,.92))
for suffix in ('png','pdf'):fig.savefig(out/f'endpoint-comparison.{suffix}',dpi=180,bbox_inches='tight')
plt.close(fig)
curves=read('curves.csv');matrices=read('matrix-trajectories.csv')
fig,axes=plt.subplots(2,3,figsize=(12,6.8))
for column,law in enumerate(laws[2:]):
    for arm,name,color in zip(arms[:2]+['swiglu143'],names[:2]+['SwiGLU143'],colors[:2]+['#4c72b0']):
        selected=sorted([r for r in curves if r['law']==law and r['arm']==arm],key=lambda r:int(r['step']))
        axes[0,column].plot([int(r['step']) for r in selected],[float(r['oracle_mse']) for r in selected],marker='.',label=name,color=color)
    axes[0,column].set_yscale('log');axes[0,column].set_title(labels[column+2]);axes[0,column].set_ylabel('Oracle field MSE')
    for layer in range(3):
        selected=sorted([r for r in matrices if r['law']==law and int(r['layer'])==layer],key=lambda r:int(r['step']))
        axes[1,column].plot([int(r['step']) for r in selected],[float(r['C_sigma_max']) for r in selected],marker='.',label=f'Block {layer+1}')
    axes[1,column].set_ylabel('Learned C largest singular value')
for ax in axes.flat:ax.set_xlabel('Training updates');ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.tight_layout()
for suffix in ('png','pdf'):fig.savefig(out/f'learning-and-generator.{suffix}',dpi=180,bbox_inches='tight')
plt.close(fig);print('Saved two PNG/PDF figures')
