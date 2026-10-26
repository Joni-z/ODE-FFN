"""Improve only cost-plot label placement, using the already audited CSVs."""
from pathlib import Path
import csv
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((root/'comparison.csv').open()))
flows=list(csv.DictReader((root/'clock-factor-trajectories.csv').open()))
matrices=list(csv.DictReader((root/'P-factor-checkpoints.csv').open()))
datasets=('gaussian_aniso_2d','gmm_ring8_2d','rotated_product_bimodal_16d')
titles=('Gaussian2D','Ring8','Product16')
arms=('vector_parent','lowrank_budget','lowrank_narrow','lowrank_augmented','swiglu','silu128','swiglu_budget','sigmoid_glu')
labels=('Dense','LR32','LR21','Aug21','Swi128','SiLU128','Swi96*','GLU128*')
colors=('#a99550','#8157a3','#548d72','#bb4242','#255f88','#669cb7','#a8c2d0','#8eaa80')
target=root/'figures/augmented-state-cost-and-extra-paths'
original=root/'qa/original-figures';original.mkdir(parents=True,exist_ok=True)
for extension in ('png','pdf'):
    saved=original/(target.name+'.'+extension)
    assert not saved.exists()
    shutil.copy2(target.with_suffix('.'+extension),saved)
fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained')
for col,(dataset,title) in enumerate(zip(datasets,titles)):
    lookup={r['label']:r for r in rows if r['dataset']==dataset};ax=axes[0,col]
    xs=[float(lookup[a]['forward_macs'])/1000 for a in arms]
    for arm,label,color,x in zip(arms,labels,colors,xs):
        y=float(lookup[arm]['validation_oracle_mse'])
        ax.scatter(x,y,s=45,color=color,marker='x' if arm in arms[-2:] else 'o',zorder=3)
        offset=(6,0)
        if dataset==datasets[2] and arm=='sigmoid_glu':offset=(6,12)
        if dataset==datasets[2] and arm=='silu128':offset=(6,-12)
        ax.annotate(label,(x,y),xytext=offset,textcoords='offset points',fontsize=9,va='center',
            arrowprops=dict(arrowstyle='-',color=color,lw=.7) if offset[1] else None)
    span=max(xs)-min(xs);ax.set_xlim(min(xs)-.05*span,max(xs)+.30*span)
    ax.margins(y=.10);ax.set_title(title);ax.set_xlabel('Dense MACs / 1000');ax.set_ylabel('Oracle MSE')
    selected=[r for r in flows if r['dataset']==dataset and r['label']=='lowrank_augmented']
    for role,key,color,label in (('lift','extra_output_rms_over_input','#bb4242','Lift extra output'),
                                  ('project','extra_input_rms_over_input','#255f88','Project extra input')):
        steps=sorted({int(r['step']) for r in selected})
        values=[np.mean([float(r[key]) for r in selected if r['role']==role and int(r['step'])==s]) for s in steps]
        axes[1,col].plot(steps,values,color=color,label=label)
    selected=[r for r in matrices if r['dataset']==dataset and r['label']=='lowrank_augmented' and r['role']=='project']
    for key,color,label in (('extra_right_columns_norm','#bb4242','V2 extra columns'),
                             ('extra_controller_input_columns_norm','#255f88','Gate2 extra input columns')):
        steps=sorted({int(r['step']) for r in selected})
        values=[np.mean([float(r[key]) for r in selected if int(r['step'])==s]) for s in steps]
        axes[2,col].plot(steps,values,color=color,label=label)
    axes[1,col].set_ylabel('Extra-feature RMS / native-input RMS')
    axes[2,col].set_ylabel('Receiving-weight Frobenius norm')
    for row in (1,2):axes[row,col].set_xlabel('Training update')
    for row in range(3):axes[row,col].grid(alpha=.15)
axes[1,2].legend(fontsize=9,frameon=False);axes[2,2].legend(fontsize=9,frameon=False)
fig.suptitle('LR32 = Factor64/q32; LR21 = Narrow64/q21; Aug21 = Expanded128/q21; * = historical control.\n'
             'Extra paths: means across three blocks, not causal importance. Dense MACs are not latency.',fontsize=10)
for extension in ('png','pdf'):fig.savefig(target.with_suffix('.'+extension),dpi=180)
plt.close(fig)
