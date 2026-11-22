"""Plot an exactly replayed Product16 trajectory and the full confirmation scores."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

doc=Path(__file__).resolve().parents[3]/'docs/report4/midpoint-refresh'
case=json.loads((doc/'escape-case.json').read_text())
results=json.loads((doc/'archived-confirmation.json').read_text())
fig,(a,b)=plt.subplots(1,2,figsize=(9.6,3.55),layout='constrained')
colors=['#a54430','#076b76','#5b598b']
labels=['Midpoint, unbounded','Midpoint, bounded','SwiGLU, unbounded']
markers=['o','s','^']
for row,label,color,marker in zip(case['rows'],labels,colors,markers):
    times=[v['t'] for v in row['trajectory']]+[1.]
    values=[v['radius'] for v in row['trajectory']]+[row['endpoint_radius']]
    a.plot(times,values,marker=marker,markersize=3,linewidth=1.65,color=color,label=label)
a.set(yscale='log',xlabel='Generation time',ylabel='Distance from origin',
      title='One shared Product16 noise vector')
a.legend(frameon=False,fontsize=8,loc='upper left')
groups=['gmm_ring8_2d','rotated_product_bimodal_16d']
x=np.arange(2)
for idx,(label,color) in enumerate(zip(['Best archived control','Bounded midpoint','Sign-equivariant SwiGLU'],['#848a8f',colors[1],colors[2]])):
    vals=[]
    for ds in groups:
        mid,=[r for r in results if r['dataset']==ds and r['family']=='bounded_odd_state_midpoint4']
        other,=[r for r in results if r['dataset']==ds and r['family'].startswith('odd_swiglu')]
        vals.append([mid['best_legacy_sw2'],mid['pooled_sw2'],other['pooled_sw2']][idx])
    bars=b.bar(x+(idx-1)*.26,vals,width=.245,label=label,color=color)
    b.bar_label(bars,fmt='%.3f',fontsize=8,padding=2)
b.set(title='Full confirmation banks',ylabel='Pooled SW2 (lower is better)',ylim=(0,.106),xticks=x,xticklabels=['Ring8','Product16'])
b.legend(frameon=False,fontsize=8,loc='upper right')
for ax in (a,b):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',linewidth=.4,alpha=.27)
    ax.set_axisbelow(True)
fig.savefig(doc/'trajectory-and-confirmation.png',dpi=180)
