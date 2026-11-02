"""Paired fixed-bank parity decomposition, separate from primary validation."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3]/'docs/autoresearch/professor-20261103/symmetry-results'
rows=list(csv.DictReader((root/'endpoints.csv').open()));profiles=list(csv.DictReader((root/'time-profiles.csv').open()))
fig,axes=plt.subplots(2,2,figsize=(12,7))
for axs,ds,title in zip(axes,['gmm_ring8_2d','rotated_product_bimodal_16d'],['Ring8','Product16']):
    rr=[r for r in rows if r['dataset']==ds];labels=['Scalar K2','Scalar K1','Raw K2','Raw K1','Swi64','Swi128']
    odd=[float(r['odd_field_mse']) for r in rr];even=[float(r['even_energy']) for r in rr]
    axs[0].bar(range(6),odd,label='Odd-part field error',color='#4477aa');axs[0].bar(range(6),even,bottom=odd,label='Even-part energy',color='#ee7733')
    axs[0].set_xticks(range(6),labels,rotation=20);axs[0].set(title=title+' — paired diagnostic',ylabel='Paired per-coordinate MSE')
    for r,name in zip(rr,labels):
        pp=[x for x in profiles if x['cell']==r['cell']]
        axs[1].plot([(float(x['t_start'])+float(x['t_end']))/2 for x in pp],[float(x['even_energy'])/float(x['paired_field_mse']) for x in pp],label=name)
    axs[1].set(title=title+' — error fraction from the even part',xlabel='Time',ylabel='Even energy / paired error',ylim=(0,1))
    for ax in axs:ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(root/f'comparison.{ext}',dpi=180,bbox_inches='tight')
