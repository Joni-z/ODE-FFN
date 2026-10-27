"""All fixed local layers, checkpoint trajectories and intervention risk accounting."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True);args=parser.parse_args()
out=args.results/'figures';out.mkdir(exist_ok=True)
def read(name):
    with (args.results/name).open() as stream:return list(csv.DictReader(stream))
local=read('local.csv');whole=read('complete.csv');parts=read('intervention-parts.csv')
families=('ode_k2','frame_k2','global_k2');labels=('Original K2','Input frame K2','Global frame K2')
cohort=[(law,family) for law in ('thin','thick') for family in families]
colors=('#666666','#c44e52','#ddaa33','#444444','#8172b3','#55a868')
names=[law.capitalize()+' / '+labels[families.index(family)] for law,family in cohort]
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(fig,name):
    for suffix in ('png','pdf'):fig.savefig(out/f'{name}.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(13,4.5))
metrics=('quadratic_relative_rms','center_box_error_relative_rms','sameclock_error_relative_rms')
titles=('Raw quadratic term / K2 increment','Best boxed clock after fixed A','Original clock after fixed A')
for ax,metric,title in zip(axes,metrics,titles):
    matrix=np.array([[100*float(r[metric]) for r in local if r['law']==law and r['family']==family
        and r['step']=='5000' and r['time_bin']=='pooled'] for law,family in cohort])
    assert matrix.shape==(6,6)
    im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=matrix.max(),aspect='auto')
    for i in range(6):
        for j in range(6):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=7,
            color='white' if matrix[i,j]>.6*matrix.max() else 'black')
    ax.set_xticks(range(6),['1.1','1.2','2.1','2.2','3.1','3.2'])
    ax.set_yticks(range(6),names if ax is axes[0] else ['']*6)
    ax.set_title(title,fontsize=9);ax.set_xlabel('Block.solve')
    fig.colorbar(im,ax=ax,shrink=.8,label='Relative RMS (%)')
fig.suptitle('Final checkpoints: all six solves; A = P + 0.3 P²',fontsize=11)
fig.tight_layout(rect=(0,0,1,.94));save(fig,'local-action')

fig,axes=plt.subplots(1,3,figsize=(12,3.7))
for (law,family),name,color in zip(cohort,names,colors):
    x=[0,1000,2500,5000]
    max_local=[max(100*float(r['center_box_error_relative_rms']) for r in local if r['law']==law
        and r['family']==family and int(r['step'])==step and r['time_bin']=='pooled') for step in x]
    values=[]
    for scope in ('full','tangent'):
        values.append([next(r['relative_displacement_rms'] for r in parts if r['law']==law
            and r['family']==family and int(r['step'])==step and r['scope']==scope) for step in x])
    for ax,ys in zip(axes,[max_local,*[[100*float(v) if v else np.nan for v in ys] for ys in values]]):
        ax.plot(x,ys,marker='o',markersize=3,color=color,label=name,
            linestyle='-' if law=='thin' else '--')
for ax,title in zip(axes,['Maximum local boxed-clock residual','Actual full-field displacement','Actual tangent displacement (reporting)']):
    ax.set_title(title,fontsize=9);ax.set_xlabel('Training update');ax.set_ylabel('Relative RMS (%)');ax.grid(alpha=.2)
    ax.set_xticks([0,2500,5000])
axes[0].axhline(1.,color='black',linestyle=':',linewidth=.8)
axes[1].axhline(1.,color='black',linestyle=':',linewidth=.8)
handles,text=axes[0].get_legend_handles_labels();fig.legend(handles,text,ncol=3,loc='upper center',frameon=False,fontsize=7)
fig.tight_layout(rect=(0,0,1,.80));save(fig,'checkpoint-response')

fig,ax=plt.subplots(figsize=(10.5,3.7));selected=[next(r for r in parts if r['law']==law and r['family']==family
    and r['step']=='5000' and r['scope']=='tangent') for law,family in cohort]
for offset,label,key,color in [(-.26,'Displacement energy','displacement_mse','#4c72b0'),
                              (0,'Error/displacement cross term','cross_term','#ddaa33'),
                              (.26,'Net risk change','net','#c44e52')]:
    values=[100*(float(r['changed_mse'])-float(r['original_mse']))/float(r['original_mse']) if key=='net'
        else 100*float(r[key])/float(r['original_mse']) for r in selected]
    ax.bar(np.arange(6)+offset,values,.24,label=label,color=color)
ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(range(6),[n.replace(' / ','\n') for n in names],fontsize=8)
ax.set_ylabel('Percent of original tangent field risk');ax.set_title('Same frozen intervention: tangent risk accounting (reporting split)')
ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout();save(fig,'risk-accounting')
print('Saved three PNG/PDF figures from every registered checkpoint/layer')
