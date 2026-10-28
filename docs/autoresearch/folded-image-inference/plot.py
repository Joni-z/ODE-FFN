"""Plot all named batch64 latency controls and measured generation paths."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out=Path(__file__).resolve().parent/'results'
rows=list(csv.DictReader((out/'timing-summary.csv').open()))
samples=json.loads((out/'generation-timings.json').read_text())
arms=('native','literal','cached','compact','swiglu129','silu193')
fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for i,(device,color) in enumerate((('cpu','#376995'),('mps','#bf6b35'))):
    values=[float(next(r for r in rows if r['device']==device and r['mode']=='resident' and r['batch']=='64' and r['arm']==a)['median_ms']) for a in arms]
    axes[0].bar(np.arange(6)+(i-.5)*.36,values,width=.36,color=color,label=device.upper())
    generation=[next(r['seconds'] for r in samples if r['device']==device and r['arm']==a) for a in ('native','cached','compact')]
    axes[1].bar(np.arange(3)+(i-.5)*.36,generation,width=.36,color=color,label=device.upper())
axes[0].set(xticks=np.arange(6),xticklabels=('Native','Literal','Cached','Compact','Swi129','SiLU193'),ylabel='Milliseconds / forward',title='Whole model, batch 64; median of 7 trials')
axes[0].tick_params(axis='x',rotation=25)
axes[1].set(xticks=np.arange(3),xticklabels=('Native','Cached','Compact'),ylabel='Seconds / 64 images',title='Heun128, one generation per setting')
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
fig.savefig(out/'inference-cost.png',dpi=180)
fig.savefig(out/'inference-cost.pdf')
