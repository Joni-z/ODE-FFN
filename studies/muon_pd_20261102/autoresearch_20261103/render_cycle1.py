"""All registered candidate endpoints, their PD parents and ordinary controls."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3]/'docs'
new=list(csv.DictReader((root/'autoresearch/professor-20261103/cycle1-results/endpoints.csv').open()))
old=[]
for f in ['research-muon-pd-2026-11-02/wave2/analysis/endpoints.csv','research-muon-pd-2026-11-02/wave2/results/endpoints.csv']:old+=list(csv.DictReader((root/f).open()))
fig,axes=plt.subplots(2,2,figsize=(10,7))
settings=[('adamw',.0003),('muon',.006),('muon',.02)]
for axs,ds in zip(axes,['gmm_ring8_2d','rotated_product_bimodal_16d']):
 for ax,key,title in zip(axs,['replayed_field_mse','numpy_test_sw2'],['Validation field MSE','Test sliced W2']):
  for offset,family,label,color in [(-.23,'vector_pd_k2','PD parent','#999999'),(0,'pd_linear_sigmoid','Full affine controller + sigmoid','#4477aa'),(.23,'pd_mlp_affine','MLP + affine output','#ee7733')]:
   values=[]
   for method,lr in settings:
    table=old if family=='vector_pd_k2' else new
    r=next(r for r in table if r['dataset']==ds and r['family']==family and r['optimizer']==method and float(r['projection_lr'])==lr)
    values.append(float(r[key]))
   ax.bar([i+offset for i in range(3)],values,width=.22,label=label,color=color)
  ordinary=min(float(r[key]) for r in old if r['dataset']==ds and r['family'].startswith('swiglu'))
  ax.axhline(ordinary,color='#228833',linestyle='--',label='Lowest ordinary control for this metric')
  ax.set_xticks(range(3),['AdamW .0003','Muon .006','Muon .02']);ax.set_title(('Ring8' if ds=='gmm_ring8_2d' else 'Product16')+' — '+title);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=9)
fig.suptitle('Cycle1: all rates and both D-construction changes; lower is better')
fig.tight_layout(rect=(0,.1,1,.96))
for suffix in ['png','pdf']:fig.savefig(root/f'autoresearch/professor-20261103/cycle1-results/comparison.{suffix}',dpi=180,bbox_inches='tight')
