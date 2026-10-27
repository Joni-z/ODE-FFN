"""All fixed arms, learning histories, coefficient spectra and generated examples."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--results',type=Path,required=True);args=parser.parse_args()
out=args.results/'figures';out.mkdir(exist_ok=True)
def read(name):
    with (args.results/name).open() as stream:return list(csv.DictReader(stream))
comparison=read('comparison.csv');curves=read('curves.csv');history=read('P-history.csv')
families=('ode_k2','jet_c','affine_c','free_c','direct_b','swiglu192','swiglu258')
names=('Original K2','Quadratic C','Affine C','Free gate C','Direct B','Swi192','Swi258')
colors=('#888888','#c44e52','#ddaa33','#55a868','#4c72b0','#8172b3','#333333')
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(fig,name):
    for suffix in ('png','pdf'):fig.savefig(out/f'{name}.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(12.5,7))
metrics=('tangent_mse_per_dimension','tangent_jacobian_error','tangent_sw2')
titles=('Tangent field MSE','Tangent-output Jacobian error','Tangent SW2')
for row,law in enumerate(('thin','thick')):
    values=[next(r for r in comparison if r['law']==law and r['family']==f) for f in families]
    for ax,metric,title in zip(axes[row],metrics,titles):
        ax.bar(range(7),[float(r[metric]) for r in values],color=colors)
        ax.set_xticks(range(7),['Orig','C','Affine','Free','B','Swi192','Swi258'],rotation=35,ha='right',fontsize=8)
        ax.set_title(law.capitalize()+' / '+title,fontsize=10);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
fig.tight_layout();save(fig,'fixed-comparison')

fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,law in zip(axes,('thin','thick')):
    for family,name,color in zip(families,names,colors):
        rows=sorted([r for r in curves if r['law']==law and r['family']==family],key=lambda r:int(r['step']))
        ax.semilogy([int(r['step']) for r in rows],[float(r['tangent_mse_per_dimension']) for r in rows],
                    marker='o',markersize=3,label=name,color=color)
    ax.set_title(law.capitalize());ax.set_xlabel('Training update');ax.set_ylabel('Tangent field MSE');ax.grid(alpha=.2)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='upper center',frameon=False,fontsize=8)
fig.tight_layout(rect=(0,0,1,.82));save(fig,'learning-curves')

fig,axes=plt.subplots(2,3,figsize=(12,6.5))
for i,law in enumerate(('thin','thick')):
    for family,name,color in zip(families[1:5],names[1:5],colors[1:5]):
        for ax,key,title in zip(axes[i],('sigma_max','C_frobenius','P_C_commutator_frobenius'),
                                ('P largest singular value','C Frobenius norm','P/C commutator Frobenius norm')):
            values=[]
            for step in (0,1000,2500,5000):
                values.append([float(r[key]) for r in history if r['law']==law and r['family']==family and int(r['step'])==step])
            array=np.array(values);assert array.shape==(4,6)
            ax.plot((0,1000,2500,5000),array.mean(1),label=name,color=color)
            ax.fill_between((0,1000,2500,5000),array.min(1),array.max(1),alpha=.12,color=color)
            ax.set_title(law.capitalize()+' / '+title,fontsize=9);ax.set_xlabel('Training update');ax.grid(alpha=.2)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='upper center',frameon=False,fontsize=8)
fig.tight_layout(rect=(0,0,1,.9));save(fig,'P-coefficient-histories')

settings=json.loads((args.root/'settings.json').read_text());reference=Path(settings['geometry_reference']);basis=np.load(args.root/'basis.npy')
fig,axes=plt.subplots(2,8,figsize=(16,4.8))
for i,law in enumerate(('thin','thick')):
    banks=Path(next(r['banks'] for r in settings['laws'] if r['law']==law))
    with np.load(banks/'sampling.npz') as data:target=data['target']@basis
    arrays=[target]
    for family in families:
        base=reference if family in ('ode_k2','swiglu192') else args.root
        with np.load(base/'output/runs'/law/(family+'_clean')/'diagnostics/samples.npz') as data:arrays.append(data['generated']@basis)
    for ax,points,label in zip(axes[i],arrays,('Target',*names)):
        ax.scatter(points[:,0],points[:,1],s=.8,alpha=.25,rasterized=True)
        ax.set_xlim(-4.5,4.5);ax.set_ylim(-4.5,4.5);ax.set_aspect('equal')
        ax.set_title(label,fontsize=8);ax.set_xticks([-3,0,3]);ax.set_yticks([-3,0,3]);ax.tick_params(labelsize=6)
    axes[i,0].set_ylabel(law.capitalize(),fontsize=9)
fig.suptitle('All 2048 saved samples projected onto the 2D target plane; normal spread is not shown',fontsize=10)
fig.tight_layout(rect=(0,0,1,.94));save(fig,'generation-cases')
print('Saved four PNG/PDF figures for every fixed arm and law')
