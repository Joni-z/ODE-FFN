"""Render completed diagnostic data only; no pending sampling scores."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
def main():
    out=HERE/'figures';out.mkdir(exist_ok=True)
    rows=json.loads((HERE/'frozen-diagnosis.json').read_text())['rows']
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,dataset,title in zip(axes,['gmm_ring8_2d','rotated_product_bimodal_16d'],['Ring8','Product16']):
        subset=[r for r in rows if r['dataset']==dataset]
        for metric,label in [('truncation_relative','Taylor8 vs exact exponential'),('restarted_composition_relative','Restarted two half steps'),('feedback_correction_relative','State evolution vs frozen')]:
            ax.plot([r['time'] for r in subset],[np.median([v[metric] for v in r['layers']]) for r in subset],marker='o',label=label)
        ax.set(title=title,xlabel='Physical FM time',ylabel='Median RMS difference / layer increment',yscale='log')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,frameon=False)
    fig.savefig(out/'composition-diagnosis.png',dpi=180);plt.close(fig)
    rows=[]
    for name in ['transport-existing.json','transport-new.json']:rows+=json.loads((HERE/name).read_text())['rows']
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,dataset,title in zip(axes,['gmm_ring8_2d','rotated_product_bimodal_16d'],['Ring8','Product16']):
        for r in rows:
            if r['dataset']!=dataset:continue
            family=r['id'].split('__')[1]
            if family not in ['screen_isru_k8','free_coeff_k8','swiglu64','swiglu192']:continue
            name={'screen_isru_k8':'Fixed exponential K8','free_coeff_k8':'Free coefficients K8','swiglu64':'SwiGLU64 / Muon','swiglu192':'SwiGLU192 / AdamW'}[family]
            ax.plot([v['weight'] for v in r['calibration']],[v['relative_rms_residual'] for v in r['calibration']],marker='o',label=name)
        ax.set(title=title,xlabel='Mixture weight toward learned field',ylabel='Relative endpoint prediction residual')
        ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False)
    fig.savefig(out/'transport-calibration.png',dpi=180);plt.close(fig)
    print(json.dumps({'passed':True,'figures':2,'output':str(out)}))

if __name__=='__main__':main()
