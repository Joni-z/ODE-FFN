"""Plot actual conditional flow strength while its eigenvalues remain fixed."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);args=p.parse_args()
root=args.results;out=root/'figures';out.mkdir(exist_ok=False)
rows=list(csv.DictReader((root/'generator-quantiles.csv').open()))
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
laws=['product','sign_code','half_mixture','gaussian_aniso_2d','gmm_ring8_2d']
labels=['Product16','Dependent signs','Half mixture','Gaussian2D','Ring8']
for law,label in zip(laws,labels):
    selected=[r for r in rows if r['family']=='swiglu128' and r['law']==law]
    steps=[0,1000,2500,5000]
    strength=[max(float(r['sigma_q100']) for r in selected if int(r['step'])==step) for step in steps]
    norms=[max(float(r['flow_norm_max']) for r in selected if int(r['step'])==step) for step in steps]
    axes[0].plot(steps,strength,marker='o',label=label);axes[1].plot(steps,norms,marker='o',label=label)
axes[0].set_ylabel('Max generator singular value on fixed probes');axes[1].set_ylabel('Max exact conditional flow norm on fixed probes')
axes[0].set_title('Strength changes with training');axes[1].set_title('All generator eigenvalues remain zero')
for ax in axes:ax.set_xlabel('Historical training updates');ax.grid(alpha=.2)
axes[0].legend(fontsize=8);axes[1].axhline(1,color='black',linestyle='--',linewidth=.8)
fig.tight_layout()
for suffix in ('png','pdf'):fig.savefig(out/f'conditional-strength.{suffix}',dpi=180,bbox_inches='tight')
plt.close(fig);print('Saved conditional strength figure')
