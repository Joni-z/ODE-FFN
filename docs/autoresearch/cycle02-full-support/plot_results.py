"""Compare all three fixed laws and show learned matrix trajectories."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();root=a.results
out=root/'figures';out.mkdir(exist_ok=False)
read=lambda name:list(csv.DictReader((root/name).open()))
rows=read('comparison.csv');laws=['product','half_mixture','sign_code'];lawlabels=['Independent signs','Half mixture','Hadamard signs']
families=['ode_k2','free_isru','swiglu64','swiglu128','swiglu160'];names=['Original ODE','Ordinary ISRU','SwiGLU64','SwiGLU128','SwiGLU160']
colors=['#999999','#c44e52','#55a868','#4c72b0','#8172b2']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(12.2,4.4))
for ax,key,title in zip(axes,['oracle_mse','jacobian_error','sw2'],['Oracle field MSE','Full network Jacobian error','Generated-sample SW2']):
    for index,(family,label,color) in enumerate(zip(families,names,colors)):
        values=[float(next(r[key] for r in rows if r['law']==law and r['family']==family)) for law in laws]
        ax.bar(np.arange(3)+(index-2)*.15,values,.14,label=label,color=color)
    ax.set_xticks(range(3),lawlabels,rotation=20,ha='right');ax.set_yscale('log');ax.set_title(title);ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Lower is better')
axes[2].set_yscale('linear');axes[2].set_ylim(0,.15)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='upper center',bbox_to_anchor=(.5,1.02),frameon=False)
fig.tight_layout(rect=(0,0,1,.94))
for suffix in ('png','pdf'):fig.savefig(out/f'law-comparison.{suffix}',dpi=180,bbox_inches='tight')
plt.close(fig)
matrices=read('matrix-trajectories.csv');curves=read('curves.csv')
fig,axes=plt.subplots(1,3,figsize=(12.2,3.8))
for ax,family,title in zip(axes[:2],['ode_k2','free_isru'],['Original ODE: learned P','Ordinary ISRU: learned maps']):
    selected=[r for r in matrices if r['family']==family]
    for layer in dict.fromkeys(r['layer'] for r in selected):
        part=sorted([r for r in selected if r['layer']==layer],key=lambda r:int(r['step']))
        label=layer.replace('raw.blocks.','block ').replace('.ffn.',' / ').replace('.proj.weight','')
        ax.plot([int(r['step']) for r in part],[float(r['sigma_max']) for r in part],marker='.',label=label)
    ax.set_title(title);ax.set_xlabel('Training updates');ax.set_ylabel('Largest singular value');ax.legend(fontsize=6.8)
for family,label,color in zip(families,names,colors):
    part=sorted([r for r in curves if r['family']==family],key=lambda r:int(r['step']))
    axes[2].plot([int(r['step']) for r in part],[float(r['oracle_mse']) for r in part],label=label,color=color,marker='.')
axes[2].set_yscale('log');axes[2].set_title('Full-mixture validation curves');axes[2].set_xlabel('Training updates');axes[2].set_ylabel('Oracle field MSE');axes[2].legend(fontsize=7)
for ax in axes:ax.grid(alpha=.2)
fig.tight_layout()
for suffix in ('png','pdf'):fig.savefig(out/f'matrices-and-learning.{suffix}',dpi=180,bbox_inches='tight')
plt.close(fig);print('Saved two PNG/PDF figures to',out)
