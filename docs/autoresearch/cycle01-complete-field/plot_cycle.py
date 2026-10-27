"""Two descriptive figures from audited outcomes; no model selection."""
from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True);args=parser.parse_args()
root=args.results;out=root/'figures';out.mkdir(exist_ok=False)
read=lambda name:list(csv.DictReader((root/name).open()))
rows=read('comparison.csv');decomposition=read('error-decomposition.csv')
datasets=['gaussian_aniso_2d','gmm_ring8_2d','rotated_product_bimodal_16d','sign_code']
labels=['Gaussian 2D','Ring8','Product16','Dependent signs']
methods=['taylor','frozen_gate','midpoint','half_euler']
names=['Complete Taylor2','Frozen-gate tangent','Midpoint','Two half Euler steps']
colors=['#c44e52','#8172b2','#dd8452','#4c72b0']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(12.3,4.3))
for ax,key,title in zip(axes,['mse_ratio','jacobian_ratio','sw2_ratio'],['Oracle field MSE','Network Jacobian error','Generated-sample SW2']):
    for index,(method,label,color) in enumerate(zip(methods,names,colors)):
        values=[float(next(r[key] for r in rows if r['dataset']==dataset and r['method']==method)) for dataset in datasets]
        ax.bar(np.arange(4)+(index-1.5)*.18,values,.17,label=label,color=color)
    ax.axhline(1,color='#333333',linewidth=1,linestyle='--')
    ax.set_yscale('log');ax.set_xticks(range(4),labels,rotation=25,ha='right');ax.set_title(title)
    ax.set_ylabel('Ratio to unchanged Euler (lower is better)');ax.grid(axis='y',alpha=.2)
handles,legend=axes[0].get_legend_handles_labels()
fig.legend(handles,legend,loc='upper center',ncol=4,bbox_to_anchor=(.5,1.03),frameon=False)
fig.tight_layout(rect=(0,0,1,.93))
for suffix in ('png','pdf'):fig.savefig(out/f'endpoint-ratios.{suffix}',dpi=180,bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4.3))
selected=[next(r for r in decomposition if r['dataset']==d and r['method']=='taylor' and r['step']=='5000' and r['position']=='all') for d in datasets]
cross=[float(r['cross_error_term']) for r in selected];energy=[float(r['displacement_energy']) for r in selected]
x=np.arange(4)
axes[0].bar(x-.18,cross,.35,label='2 mean(error × displacement)',color='#55a868')
axes[0].bar(x+.18,energy,.35,label='mean(displacement²)',color='#c44e52')
axes[0].axhline(0,color='black',linewidth=.7);axes[0].set_ylabel('Contribution to field MSE change')
axes[0].set_title('Exact output-error decomposition');axes[0].legend(fontsize=8)
positions=['1','2','3','all']
for i,dataset in enumerate(datasets):
    source=[r for r in read('field.csv') if r['dataset']==dataset and r['step']=='5000']
    baseline=float(next(r['oracle_mse'] for r in source if r['method']=='euler'))
    values=[float(next(r['oracle_mse'] for r in source if r['method']=='taylor' and r['position']==p))/baseline for p in positions]
    axes[1].plot(range(4),values,marker='o',label=labels[i])
axes[1].set_yscale('log');axes[1].axhline(1,color='black',linewidth=.7,linestyle='--')
axes[1].set_xticks(range(4),['Block 1','Block 2','Block 3','All blocks'])
axes[1].set_ylabel('Field MSE / unchanged Euler');axes[1].set_title('Locations are diagnostic, not selected');axes[1].legend(fontsize=8)
axes[0].set_xticks(x,labels,rotation=25,ha='right')
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.tight_layout()
for suffix in ('png','pdf'):fig.savefig(out/f'error-decomposition.{suffix}',dpi=180,bbox_inches='tight')
plt.close(fig)
print('Saved two PNG/PDF figures to',out)
