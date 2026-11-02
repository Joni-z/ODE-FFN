"""Descriptive decomposition, averaged over six matrices; not causal attribution."""
import csv,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3]/'docs/autoresearch/professor-20261103/spectrum'
rows=list(csv.DictReader((root/'row-column.csv').open()))
fig,axs=plt.subplots(2,3,figsize=(11,6))
for axes,ds in zip(axs,['gmm_ring8_2d','rotated_product_bimodal_16d']):
 for method,lr,color in [('adamw','0.0003','#4477aa'),('adamw','0.001','#66ccee'),('muon','0.006','#ee7733'),('muon','0.02','#cc3311')]:
  sub=[r for r in rows if r['dataset']==ds and r['optimizer']==method and r['lr']==lr];steps=sorted({int(r['step']) for r in sub})
  for ax,key in zip(axes,['sigma_max','row_norm_mean','row_normalized_sigma']):
   means=[statistics.mean(float(r[key]) for r in sub if int(r['step'])==s) for s in steps]
   ax.plot([s/1000 for s in steps],means,color=color,label=f'{method} {lr}');ax.set_xlabel('Updates (thousands)');ax.grid(alpha=.2)
 axes[0].set_ylabel('Ring8' if ds=='gmm_ring8_2d' else 'Product16')
for ax,title in zip(axs[0],['Mean P spectral norm','Mean row magnitude','Mean spectral norm after row normalization']):ax.set_title(title)
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4)
fig.suptitle('Archived scalar ODE: scale growth versus row-direction coherence')
fig.tight_layout(rect=(0,.06,1,.96))
for ext in ['png','pdf']:fig.savefig(root/f'row-decomposition.{ext}',dpi=180)
