"""Full cycle2 endpoint matrix with cycle1 references and ordinary controls."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3]/'docs';study=root/'autoresearch/professor-20261103'
new=list(csv.DictReader((study/'cycle2-results/endpoints.csv').open()));first=list(csv.DictReader((study/'cycle1-results/endpoints.csv').open()));old=[]
for f in ['research-muon-pd-2026-11-02/wave2/analysis/endpoints.csv','research-muon-pd-2026-11-02/wave2/results/endpoints.csv']:old+=list(csv.DictReader((root/f).open()))
settings=[('adamw',.0003),('muon',.006),('muon',.02)]
fig,axes=plt.subplots(2,2,figsize=(12,8))
for axs,ds in zip(axes,['gmm_ring8_2d','rotated_product_bimodal_16d']):
 for ax,key,title in zip(axs,['replayed_field_mse','numpy_test_sw2'],['Validation field MSE','Test sliced W2']):
  for offset,family,label,color,table in [(-.32,'vector_pd_k2','MLP PD parent','#999999',old),(-.16,'pd_linear_sigmoid','Full affine controller','#66ccee',first),(0,'pd_factorized_linear','Factorized linear controller','#4477aa',new),(.16,'pd_direct_raw','Direct h diagonal','#ee7733',new),(.32,'pd_direct_bounded','Direct bounded diagonal','#aa3377',new)]:
   rr=[next(r for r in table if r['dataset']==ds and r['family']==family and r['optimizer']==method and float(r['projection_lr'])==lr) for method,lr in settings]
   ax.bar([i+offset for i in range(3)],[float(r[key]) for r in rr],width=.15,label=label,color=color)
  ordinary=min(float(r[key]) for r in old if r['dataset']==ds and r['family'].startswith('swiglu'))
  ax.axhline(ordinary,color='#228833',linestyle='--',label='Lowest ordinary control for this metric')
  ax.set_xticks(range(3),['AdamW .0003','Muon .006','Muon .02']);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.set_title(('Ring8' if ds=='gmm_ring8_2d' else 'Product16')+' — '+title)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
fig.suptitle('Cycle2: rank/factorization and direct input diagonals; all registered rates')
fig.tight_layout(rect=(0,.1,1,.96))
for ext in ['png','pdf']:fig.savefig(study/f'cycle2-results/comparison.{ext}',dpi=180,bbox_inches='tight')
