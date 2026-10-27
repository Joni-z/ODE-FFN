"""Display signed full-bank curvature effects without selecting a coefficient."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True);args=parser.parse_args();root=args.results
with (root/'field-attribution.csv').open() as stream:rows=list(csv.DictReader(stream))
labels=['Gaussian2D','Ring8','Product16','Dependent signs','Half mixture'];x=np.arange(5)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
for key,label,color,offset in [('alignment_cross_term','Signed alignment','#4c72b0',-.16),('displacement_energy','Displacement energy','#dd8452',.16)]:
    values=[100*float(r[key])/float(r['midpoint_counterfactual_MSE']) for r in rows]
    axes[0].bar(x+offset,values,.30,label=label,color=color)
axes[0].set_title('Exact squared-error decomposition');axes[0].legend(frameon=False,fontsize=8)
values=[100*float(r['actual_minus_counterfactual_MSE'])/float(r['midpoint_counterfactual_MSE']) for r in rows]
axes[1].bar(x,values,.6,color=['#55a868' if v<0 else '#c44e52' for v in values])
axes[1].set_title('Net effect of retaining nonlinear curvature')
for ax in axes:
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(x,labels,rotation=25,ha='right')
    ax.set_ylabel('Change / midpoint counterfactual MSE (%)');ax.grid(axis='y',alpha=.2)
fig.tight_layout();out=root/'figures';out.mkdir(exist_ok=False)
for suffix in ('png','pdf'):fig.savefig(out/f'curvature-attribution.{suffix}',dpi=180,bbox_inches='tight')
plt.close(fig)
