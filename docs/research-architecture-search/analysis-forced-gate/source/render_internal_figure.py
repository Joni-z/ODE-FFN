"""Re-render the audited internal trajectories with readable wrapped labels."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
out=Path(__file__).resolve().parents[1]
flows=list(csv.DictReader((out/'forced-gate-trajectories.csv').open()))
projections=list(csv.DictReader((out/'projection-trajectories.csv').open()))
arms=('swiglu_budget','swiglu','forced_static','forced_k2','forced_exact')
labels=('SwiGLU96','SwiGLU128','Static response','Adaptive K2','Adaptive exact')
colors=('#7ba8bd','#255f88','#9e7b53','#bb4242','#923b8f')
tasks=('gaussian_aniso_2d','gmm_ring8_2d','rotated_product_bimodal_16d')
fig,axes=plt.subplots(3,3,figsize=(12,9),layout='constrained')
for col,(dataset,title) in enumerate(zip(tasks,('Gaussian2D','Ring8','Product16'))):
 for row,(metric,name) in enumerate((('P_sigma_max','Max |lambda|\nDotted: conditional rate'),('correction_rms_over_baseline','Correction RMS /\nbaseline gate RMS'),('gate_value_output_norm_product','Gate/value/output\nspectral-norm product'))):
  ax=axes[row,col];entries=projections if row==2 else flows
  for i in (range(5) if row==2 else range(2,5)):
   trace=[r for r in entries if r['dataset']==dataset and r['label']==arms[i]];steps=sorted({int(r['step']) for r in trace})
   values=np.array([[float(r[metric]) for r in trace if int(r['step'])==s] for s in steps])
   ax.plot(steps,values.mean(1),label=labels[i],color=colors[i]);ax.fill_between(steps,values.min(1),values.max(1),color=colors[i],alpha=.09)
   if row==0:
    rates=np.array([[float(r['conditional_rate_abs_max']) for r in trace if int(r['step'])==s] for s in steps]);ax.plot(steps,rates.mean(1),color=colors[i],ls=':',lw=1)
  ax.set(title=title if row==0 else '',xlabel='Training update',ylabel=name);ax.grid(alpha=.15)
axes[0,-1].legend(fontsize=7,frameon=False);axes[2,-1].legend(fontsize=7,frameon=False)
fig.suptitle('Lines: three-block mean; shading: block range, not uncertainty. No full-network stability bound.',fontsize=9)
for extension in ('png','pdf'):fig.savefig(out/'figures'/f'forced-gate-internals.{extension}',dpi=180)
