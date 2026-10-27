"""Visualize all contexts, training curves, original P histories and samples."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--results',type=Path,required=True)
args=parser.parse_args();root=args.root;results=args.results;out=results/'figures';out.mkdir(exist_ok=True)
read=lambda name:list(csv.DictReader((results/name).open()))
rows=read('comparison.csv');curves=read('curves.csv');ps=read('P-trajectories.csv')
families=['ode_k2','swiglu128','swiglu192'];names=['Original ODE K2','Swi128 (parameter control)','Swi192 (compute control)'];colors=['#c44e52','#55a868','#4c72b0']
heads=['clean','shifted','direct'];laws=['thin','thick']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})


def save(fig,name):
    for suffix in ('png','pdf'):fig.savefig(out/f'{name}.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)


fig,axes=plt.subplots(2,3,figsize=(12,6.4))
for i,law in enumerate(laws):
    for j,(key,label) in enumerate([('oracle_mse','Full128D field MSE'),('tangent_mse_per_dimension','Tangent field MSE / 2D'),('normal_mse_per_dimension','Normal field MSE / 126D')]):
        ax=axes[i,j]
        for k,(family,name,color) in enumerate(zip(families,names,colors)):
            values=[float(next(r[key] for r in rows if r['law']==law and r['family']==family and r['head']==head)) for head in heads]
            ax.bar(np.arange(3)+(k-1)*.24,values,.22,label=name,color=color)
        ax.set_yscale('log');ax.set_xticks(range(3),['Clean','Shifted','Direct']);ax.set_title(f'{law.capitalize()}: {label}');ax.grid(axis='y',alpha=.2)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='upper center',frameon=False)
fig.tight_layout(rect=(0,0,1,.94));save(fig,'head-and-geometry')
fig,axes=plt.subplots(2,3,figsize=(12,6.4))
for i,law in enumerate(laws):
    for j,head in enumerate(heads):
        ax=axes[i,j]
        for family,name,color in zip(families,names,colors):
            selected=sorted([r for r in curves if r['law']==law and r['family']==family and r['head']==head],key=lambda r:int(r['step']))
            ax.plot([int(r['step']) for r in selected],[float(r['oracle_mse']) for r in selected],label=name,color=color,marker='.')
        ax.set_yscale('log');ax.set_title(f'{law.capitalize()} / {head}');ax.set_xlabel('Updates');ax.set_ylabel('Decoded-field oracle MSE');ax.grid(alpha=.2)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='upper center',frameon=False)
fig.tight_layout(rect=(0,0,1,.94));save(fig,'learning-curves')
fig,axes=plt.subplots(2,3,figsize=(12,6.4))
layers=sorted({r['layer'] for r in ps})
for i,law in enumerate(laws):
    for j,head in enumerate(heads):
        ax=axes[i,j]
        for layer in layers:
            selected=sorted([r for r in ps if r['law']==law and r['head']==head and r['layer']==layer],key=lambda r:int(r['step']))
            ax.plot([int(r['step']) for r in selected],[float(r['sigma_max']) for r in selected],label=layer.replace('blocks.','B').replace('.ffn.ode','/'),marker='.')
        ax.set_title(f'Original ODE: {law} / {head}');ax.set_xlabel('Updates');ax.set_ylabel('P largest singular value');ax.grid(alpha=.2)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=6,loc='upper center',frameon=False)
fig.tight_layout(rect=(0,0,1,.94));save(fig,'P-trajectories')
basis=np.load(root/'basis.npy');columns=[('Target',None,None),('Oracle Heun32',None,'oracle')]+[(f'{family} / {head}',family,head) for head in heads for family in ('ode_k2','swiglu192')]
fig,axes=plt.subplots(2,len(columns),figsize=(16,5.6))
for i,law in enumerate(laws):
    for j,(title,family,head) in enumerate(columns):
        if head is None:
            with np.load(root/'banks'/law/'sampling.npz') as data:points=data['target']
        elif head=='oracle':
            with np.load(root/'banks'/law/'oracle-samples.npz') as data:points=data['generated']
        else:
            with np.load(root/'output/runs'/law/f'{family}_{head}'/'diagnostics/samples.npz') as data:points=data['generated']
        plane=points@basis;normal=points-plane@basis.T;rms=float(np.sqrt(np.sum(normal**2)/(len(points)*126)))
        ax=axes[i,j];ax.scatter(plane[:,0],plane[:,1],s=1.3,alpha=.28,rasterized=True)
        ax.set_xlim(-4.5,4.5);ax.set_ylim(-4.5,4.5);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
        if i==0:ax.set_title(title.replace('ode_k2','ODE').replace('swiglu192','Swi192'),fontsize=8)
        if j==0:ax.set_ylabel(law.capitalize())
        ax.set_xlabel(f'Normal RMS {rms:.3f}',fontsize=8)
fig.suptitle('Evaluation-only projection onto latent plane; normal RMS exposes off-plane error',fontsize=10)
fig.subplots_adjust(left=.025,right=.995,bottom=.07,top=.88,wspace=.12,hspace=.35);save(fig,'generation-cases')
print('Saved four PNG/PDF figures')
