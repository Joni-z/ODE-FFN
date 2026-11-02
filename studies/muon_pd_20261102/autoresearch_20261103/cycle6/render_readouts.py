"""Plot all registered final80k examples; no selection or model execution."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--readouts',nargs=2,required=True)
p.add_argument('--out',required=True)
a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
data=[json.loads(Path(f).read_text()) for f in a.readouts]
assert all(len(d['rows'])==4 for d in data)
product=next(d for d in data if d['reference'] is not None)
ring=next(d for d in data if d['reference'] is None)
families=['pd_direct_raw_k1','swiglu64','swiglu128','swiglu192']
labels=['Raw K1','SwiGLU64','SwiGLU128','SwiGLU192']
rows=[r for d in data for r in d['rows']]
keys=[k for k in dict.fromkeys(k for r in rows for k in r) if k!='axis_gap_mass']
with (out/'sample-readouts.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
with (out/'sample-readouts.json').open('w') as f:json.dump(data,f,indent=2);f.write('\n')
fig,axes=plt.subplots(4,3,figsize=(11,10))
ref=product['reference']
for axs,family,label in zip(axes,families,labels):
    for ax,j in zip(axs,[0,7,15]):
        e=next(e for e in product['examples'] if e['family']==family and e['axis']==j)
        edges=np.array(e['edges']);width=np.diff(edges)
        ax.stairs(np.array(e['counts'])/e['sample_count']/width,edges,label=label,color='#4477aa')
        mean=ref['locations'][j];var=ref['variances'][j];x=np.linspace(edges[0],edges[-1],1000)
        pdf=(np.exp(-(x-mean)**2/(2*var))+np.exp(-(x+mean)**2/(2*var)))/(2*np.sqrt(2*np.pi*var))
        ax.plot(x,pdf,color='black',lw=1,label='Population')
        row=next(r for r in product['rows'] if r['family']==family)
        ax.set_title(f'{label}, axis {j}: gap {100*row["axis_gap_mass"][j]:.2f}% / {100*ref["population_gap_mass"][j]:.2f}% true',fontsize=9)
        ax.set(xlabel='Latent coordinate',ylabel='Density');ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'product-examples.{ext}',dpi=160,bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(1,5,figsize=(15,3),sharex=True,sharey=True)
target=np.array(ring['examples'][0]['target']);axes[0].scatter(*target.T,s=2,alpha=.5,color='black');axes[0].set_title('Target (same indices)')
for ax,family,label in zip(axes[1:],families,labels):
    e=next(e for e in ring['examples'] if e['family']==family);x=np.array(e['generated'])
    assert len(x)==1024
    ax.scatter(*x.T,s=2,alpha=.5);ax.set_title(label)
for ax in axes:ax.set(xlim=(-4,4),ylim=(-4,4),aspect='equal');ax.grid(alpha=.15)
fig.suptitle('Ring8 final80k: fixed 1024/4096 display indices, no points outside frame')
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'ring-examples.{ext}',dpi=160,bbox_inches='tight')
print(json.dumps(dict(rows=len(rows),figures=2,save=str(out))))
