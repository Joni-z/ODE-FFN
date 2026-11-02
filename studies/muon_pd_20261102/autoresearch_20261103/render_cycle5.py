"""Matched-spectrum feature contrast with all ordinary rates and parity readout."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3]/'docs';study=root/'autoresearch/professor-20261103'
tables={i:list(csv.DictReader((study/f'cycle{i}-results/endpoints.csv').open())) for i in [3,4,5]};old=[]
for file in ['research-muon-pd-2026-11-02/wave2/analysis/endpoints.csv','research-muon-pd-2026-11-02/wave2/results/endpoints.csv']:
    old+=list(csv.DictReader((root/file).open()))
settings=[('adamw',.0003),('adamw',.001),('muon',.006),('muon',.02)]
fig,axes=plt.subplots(2,2,figsize=(12,8))
designs=[(3,'pd_direct_raw_k1','Raw K1',-.3,'#eeaa33'),(5,'pd_direct_abs_k1','Abs K1',-.15,'#ee7733'),(5,'bilinear64','Bilinear64',0,'#66ccee'),(3,'swiglu64','SwiGLU64',.15,'#228833'),(4,'pd_constant_k1','Constant ordinary',.3,'#4477aa')]
for axs,ds in zip(axes,['gmm_ring8_2d','rotated_product_bimodal_16d']):
    for ax,key,title in zip(axs,['replayed_field_mse','numpy_test_sw2'],['Validation field MSE','Test sliced W2']):
        for cycle,family,label,offset,color in designs:
            for i,(method,lr) in enumerate(settings):
                matches=[r for r in tables[cycle] if r['dataset']==ds and r['family']==family and r['optimizer']==method and float(r['projection_lr'])==lr]
                if matches:ax.bar(i+offset,float(matches[0][key]),width=.14,color=color,label=label if i==0 else None)
        ordinary=min(float(r[key]) for r in old+tables[3] if r['dataset']==ds and r['family'].startswith('swiglu'))
        ax.axhline(ordinary,color='#228833',ls='--',label='Lowest SwiGLU result for this metric')
        ax.set_xticks(range(4),['Adam .0003','Adam .001','Muon .006','Muon .02']);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
        ax.set_title(('Ring8' if ds=='gmm_ring8_2d' else 'Product16')+' — '+title)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
fig.suptitle('Cycle5: raw versus absolute diagonal, with ordinary bilinear control')
fig.tight_layout(rect=(0,.1,1,.96))
for ext in ['png','pdf']:fig.savefig(study/f'cycle5-results/comparison.{ext}',dpi=180,bbox_inches='tight')
parity=study/'cycle5-results/parity/endpoints.csv'
if parity.exists():
    previous=list(csv.DictReader((study/'symmetry-results/endpoints.csv').open()));current=list(csv.DictReader(parity.open()))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,ds,title in zip(axes,['gmm_ring8_2d','rotated_product_bimodal_16d'],['Ring8','Product16']):
        choose=[next(r for r in previous if r['dataset']==ds and '__pd_direct_raw_k1__' in r['cell'])]+[next(r for r in current if r['dataset']==ds and f'__{family}__' in r['cell']) for family in ['pd_direct_abs_k1','bilinear64']]+[next(r for r in previous if r['dataset']==ds and f'__{family}__' in r['cell']) for family in ['swiglu64','swiglu128']]
        odd=[float(r['odd_field_mse']) for r in choose];even=[float(r['even_energy']) for r in choose]
        ax.bar(range(5),odd,color='#4477aa',label='Odd-part field error');ax.bar(range(5),even,bottom=odd,color='#ee7733',label='Even-part energy')
        ax.set_xticks(range(5),['Raw K1','Abs K1','Bilinear64','Swi64','Swi128'],rotation=20);ax.set(title=title+' — validation-selected family checkpoints',ylabel='Paired diagnostic MSE');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(study/f'cycle5-results/parity-comparison.{ext}',dpi=180,bbox_inches='tight')
