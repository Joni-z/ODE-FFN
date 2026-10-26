"""Render the completed composition diagnostics; no model evaluation."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def main(raw,output):
    assert json.loads((raw/'status.json').read_text())['state']=='COMPLETE'
    assert json.loads((raw/'audit.json').read_text())['status']=='PASS'
    output.mkdir(parents=True,exist_ok=False)
    for name in ('partial-field-errors.csv','component-energies.csv','partial-jacobian-errors.csv',
                 'component-skew.csv','finite-differences.csv','complete-gradient-checks.csv',
                 'component-grams.json','input-bindings.json','audit.json'):
        shutil.copy2(raw/name,output/name)
    rows=list(csv.DictReader((raw/'partial-field-errors.csv').open()))
    families=('ode_k2','swiglu128','swiglu192')
    datasets=('gaussian_aniso_2d','gmm_ring8_2d','rotated_product_bimodal_16d')
    cases=[(family,trained) for family in families for trained in ('gradient','directional')]
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.8),sharey=True)
    for ax,dataset,title in zip(axes,datasets,('Gaussian2D','Ring8','Product16')):
        matrix=np.array([[next(float(r['oracle_mse'])/float(r['full_oracle_mse']) for r in rows if
            (r['dataset'],r['family'],r['trained'],r['maximum_degree'])==(dataset,family,trained,str(k)))
            for k in range(4)] for family,trained in cases])
        picture=ax.imshow(np.log10(matrix),aspect='auto',cmap='coolwarm',
            norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=2))
        for i in range(6):
            for j in range(4):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=9)
        ax.set(title=title,xticks=range(4),xticklabels=['0','0–1','0–2','0–3'],
               yticks=range(6),yticklabels=[f'{f} / {t}' for f,t in cases],xlabel='Included degrees')
    fig.subplots_adjust(left=.22,right=.88,wspace=.13,bottom=.15,top=.9)
    bar=fig.add_axes([.905,.18,.015,.65])
    fig.colorbar(picture,cax=bar,label='log10(partial/full field MSE)')
    for suffix in ('png','pdf'):fig.savefig(output/f'partial-field-errors.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    skew=list(csv.DictReader((raw/'component-skew.csv').open()))
    fig,axes=plt.subplots(1,3,figsize=(11,3.3))
    for ax,family in zip(axes,families):
        rr=[r for r in skew if (r['dataset'],r['family'],r['trained'],r['time'])==(datasets[2],family,'gradient','0.5')]
        rr=sorted(rr,key=lambda r:int(r['degree']))
        full=float(rr[0]['full_skew_energy'])
        values=[float(r['component_skew_energy']) for r in rr]+[full]
        ax.bar(range(5),values)
        ax.set(title=family,xticks=range(5),xticklabels=['C0','C1','C2','C3','sum'])
        ax.text(.98,.93,f'Full: {full:.2e}',transform=ax.transAxes,ha='right',va='top',fontsize=8)
    axes[0].set_ylabel('Mean squared Frobenius skew norm')
    fig.suptitle('Product16, t=0.5, A-trained weights',fontsize=10)
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(output/f'skew-cancellation.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(dict(status='PASS',report_figures=2,output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.raw.resolve(),a.output.resolve())
