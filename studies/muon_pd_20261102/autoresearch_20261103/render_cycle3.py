"""Registered order contrasts, complete rate matrix and frozen solver profiles."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3]/'docs';study=root/'autoresearch/professor-20261103'
new=list(csv.DictReader((study/'cycle3-results/endpoints.csv').open()))
two=list(csv.DictReader((study/'cycle2-results/endpoints.csv').open()));old=[]
for file in ['research-muon-pd-2026-11-02/wave2/analysis/endpoints.csv','research-muon-pd-2026-11-02/wave2/results/endpoints.csv']:
    old+=list(csv.DictReader((root/file).open()))
settings=[('adamw',.0003),('adamw',.001),('muon',.006),('muon',.02)]
fig,axes=plt.subplots(2,2,figsize=(12,8))
for axs,ds in zip(axes,['gmm_ring8_2d','rotated_product_bimodal_16d']):
    for ax,key,title in zip(axs,['replayed_field_mse','numpy_test_sw2'],['Validation field MSE','Test sliced W2']):
        for table,family,label,offset,color in [(old,'scalar_k2','Scalar K2',-.3,'#999999'),(new,'scalar_k1','Scalar K1',-.15,'#4477aa'),(two,'pd_direct_raw','Direct h K2',0,'#eeaa33'),(new,'pd_direct_raw_k1','Direct h K1',.15,'#ee7733'),(new,'swiglu64','Native SwiGLU64',.3,'#228833')]:
            for i,(method,lr) in enumerate(settings):
                matches=[r for r in table if r['dataset']==ds and r['family']==family and r['optimizer']==method and float(r['projection_lr'])==lr]
                if matches:ax.bar(i+offset,float(matches[0][key]),width=.14,color=color,label=label if i==0 else None)
        ordinary=min(float(r[key]) for r in old+new if r['dataset']==ds and r['family'].startswith('swiglu'))
        ax.axhline(ordinary,color='#228833',ls='--',label='Lowest ordinary result for this metric')
        ax.set_xticks(range(4),['Adam .0003','Adam .001','Muon .006','Muon .02']);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
        ax.set_title(('Ring8' if ds=='gmm_ring8_2d' else 'Product16')+' — '+title)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
fig.suptitle('Cycle3: remove the tied second-order term; retain ordinary budget controls')
fig.tight_layout(rect=(0,.1,1,.96))
for ext in ['png','pdf']:fig.savefig(study/f'cycle3-results/comparison.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
solver=study/'solver-results'
if solver.exists():
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for axs,host,label in zip(axes,['lenovo','145'],['Ring8','Product16']):
        receipt=json.loads((solver/f'{host}.json').read_text())
        for row,name,color in zip(receipt['rows'],['Direct h K2','Scalar K2','SwiGLU128'],['#ee7733','#4477aa','#228833']):
            axs[0].plot([r['nfe'] for r in row['sampling']],[r['numpy_test_sw2'] for r in row['sampling']],marker='o',label=name,color=color)
            axs[1].plot([(r['t_start']+r['t_end'])/2 for r in row['time_profile']],[r['field_mse'] for r in row['time_profile']],marker='.',label=name,color=color)
        axs[0].axhline(receipt['rows'][0]['sampling_floor'],ls=':',c='gray',label='Independent target-target distance')
        axs[0].set(title=label+' — frozen solver resolution',xlabel='Field evaluations (NFE)',ylabel='Test sliced W2')
        axs[1].set(title=label+' — fixed validation time profile',xlabel='Time',ylabel='Oracle field MSE')
        for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(solver/f'profiles.{ext}',dpi=180,bbox_inches='tight')
