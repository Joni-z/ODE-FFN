"""Audited temporal contributions, shared normal restriction and fixed cases."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--results',type=Path,required=True);args = parser.parse_args()
out = args.results/'figures';out.mkdir(exist_ok=True)
def read(name):
    with (args.results/name).open() as stream:return list(csv.DictReader(stream))
endpoints,steps,bins = [read(name) for name in ('endpoints.csv','steps.csv','bins.csv')]
settings = json.loads((args.root/'settings.json').read_text());basis = np.load(Path(settings['parent'])/'basis.npy')
families = ['ode_k2','frame_k2','global_k2','frame_k1','swiglu192','swiglu257']
names = ['Original K2','Input frame K2','Global frame K2','Input frame K1','Swi192','Swi257']
colors = ['#888888','#c44e52','#ddaa33','#8172b3','#55a868','#4c72b0']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(fig,name):
    for suffix in ('png','pdf'):fig.savefig(out/f'{name}.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)
def select_steps(law,family,scope):
    return sorted([r for r in steps if r['law']==law and r['family']==family and r['scope']==scope],key=lambda r:int(r['step']))

fig,axes = plt.subplots(1,2,figsize=(11.8,4.5))
matrices = []
for law in ('thin','thick'):
    matrices.append(np.array([[100*float(next(r['signed_share'] for r in bins if r['law']==law and r['family']==family
        and r['scope']=='tangent' and r['time_bin']==f'{i/4:.2f}-{(i+1)/4:.2f}')) for i in range(4)] for family in families]))
bound = max(abs(m).max() for m in matrices)
for ax,law,matrix in zip(axes,('thin','thick'),matrices):
    im = ax.imshow(matrix,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
    ax.set_yticks(range(6),names);ax.set_xticks(range(4),['0–.25','.25–.5','.5–.75','.75–1'])
    for i in range(6):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]:.1f}%',ha='center',va='center',color='white' if abs(matrix[i,j])>.65*bound else 'black',fontsize=8)
    ax.set_title(f'{law.capitalize()}: signed tangent endpoint accounting');ax.set_xlabel('Model step-start time bin')
fig.subplots_adjust(left=.12,right=.89,bottom=.16,top=.88,wspace=.43)
cax = fig.add_axes([.92,.17,.015,.68]);fig.colorbar(im,cax=cax,label='Signed fraction of paired endpoint MSE (%)')
save(fig,'temporal-accounting')

fig,axes = plt.subplots(1,2,figsize=(11.5,4))
thick = [next(r for r in endpoints if r['law']=='thick' and r['family']==f) for f in families]
forced = [float(r['forced_normal_endpoint_mse']) for r in thick]
remaining = [float(r['normal_paired_endpoint_mse'])-f for r,f in zip(thick,forced)]
axes[0].bar(range(6),forced,color='#4c72b0',label='Fixed-readout normal subspace')
axes[0].bar(range(6),remaining,bottom=forced,color='#ddaa33',label='Orthogonal normal remainder')
axes[0].set_xticks(range(6),['Orig','Frame\nK2','Global\nK2','Frame\nK1','Swi192','Swi257'])
axes[0].set_ylabel('Paired endpoint MSE per normal coordinate');axes[0].set_title('Thick: 91.5–93.2% lies in the forced subspace')
axes[0].legend(fontsize=8,frameon=False,loc='lower right')
for family,name,color in zip(families,names,colors):
    rows = select_steps('thick',family,'full');end = next(r for r in endpoints if r['law']=='thick' and r['family']==family)
    axes[1].plot([float(r['time']) for r in rows]+[1.],[float(r['model_path_normal_rms']) for r in rows]+[float(end['model_endpoint_normal_rms'])],color=color,label=name)
rows = select_steps('thick','frame_k2','full');end = next(r for r in endpoints if r['law']=='thick' and r['family']=='frame_k2')
axes[1].plot([float(r['time']) for r in rows]+[1.],[float(r['oracle_path_normal_rms']) for r in rows]+[float(end['oracle_endpoint_normal_rms'])],color='black',linestyle='--',label='Fixed oracle Heun32')
axes[1].set_xlabel('External time');axes[1].set_ylabel('Normal RMS, 126-coordinate normalization')
axes[1].set_title('All fitted paths lose too much normal variation')
axes[1].legend(fontsize=7,frameon=False)
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.tight_layout();save(fig,'normal-restriction')

fig,axes = plt.subplots(1,2,figsize=(11.4,3.8))
for law,ax in zip(('thin','thick'),axes):
    for family,name,color in zip(families,names,colors):
        rows = select_steps(law,family,'tangent')
        ax.plot([float(r['time']) for r in rows],[float(r['finite_rms_gain']) if r['finite_rms_gain'] else np.nan for r in rows],color=color,label=name)
    ax.set_xlabel('External step-start time');ax.set_ylabel('Transported/local tangent defect RMS ratio')
    ax.set_title(law.capitalize());ax.grid(alpha=.2)
handles,labels = axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='upper center',frameon=False,fontsize=8)
fig.tight_layout(rect=(0,0,1,.84));save(fig,'finite-defect-gain')

indices = [0,512,1024,1536];fig,axes = plt.subplots(2,4,figsize=(12,6.6))
means = 3*np.column_stack((np.cos(np.arange(8)*2*np.pi/8),np.sin(np.arange(8)*2*np.pi/8)))
for i,law in enumerate(('thin','thick')):
    with np.load(args.root/'output/raw'/f'{law}-oracle-path.npz') as data:reference = data['path'][:,indices]@basis
    for j,index in enumerate(indices):
        ax = axes[i,j];ax.scatter(means[:,0],means[:,1],color='#cccccc',s=12,marker='+')
        for family,name,color in zip(families,names,colors):
            with np.load(args.root/'output/raw'/f'{law}-{family}.npz') as data:plane = data['path'][:,index]@basis
            ax.plot(plane[:,0],plane[:,1],color=color,label=name,linewidth=1.1)
            ax.scatter(plane[-1,0],plane[-1,1],color=color,s=10)
        ax.plot(reference[:,j,0],reference[:,j,1],color='black',linestyle='--',label='Oracle',linewidth=1.2)
        ax.set_xlim(-4.5,4.5);ax.set_ylim(-4.5,4.5);ax.set_aspect('equal');ax.set_title(f'{law.capitalize()} / source {index}',fontsize=9)
        ax.set_xticks([-3,0,3]);ax.set_yticks([-3,0,3])
handles,labels = axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='upper center',fontsize=8,frameon=False)
fig.tight_layout(rect=(0,0,1,.9));save(fig,'fixed-source-paths')
print('Saved four PNG/PDF figures from audited fixed-path evidence')
