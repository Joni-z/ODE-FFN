"""Render only the independently completed25-case numerical cohort."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--portable',required=True);a=p.parse_args()
    study=Path(a.study);portable=Path(a.portable);verification=json.loads((study/'numerical-completion.json').read_text())
    assert verification['passed'] and verification['cases']==25 and verification['rows']==78
    assert sha(portable/'portable.json')==verification['portable_manifest_sha256']
    cases=json.loads((portable/'portable.json').read_text())['cases'];sym=json.loads((study/'symmetry-full.json').read_text());assert sym['passed'] and len(sym['cases'])==7
    with (study/'numerical-scores.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==78
    families=['screen_isru_k1','screen_isru_k8','free_coeff_k8','state_midpoint4','free_stage_midpoint4','free_isru','swiglu64']
    labels=['K1','K8','Free coeff.','State flow','Free stages','Free ISRU','SwiGLU']
    selected=[next(c for c in sym['cases'] if c['family']==family) for family in families]
    figdir=study/'figures';outputs=[figdir/'symmetry-full.png',figdir/'internal-resolution-full.png'];assert not any(p.exists() for p in outputs)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    f,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,key,label in zip(axes,['stratified_sw2','sector_l1'],['C8-stratified SW2','Sector L1 imbalance']):
        for mode,offset,color in [('original',-.18,'#0072B2'),('projected',.18,'#D55E00')]:
            values=[next(r for r in c['rows'] if r['mode']==mode)[key] for c in selected]
            ax.bar(np.arange(7)+offset,values,.36,label=mode.capitalize(),color=color)
        ax.set(xticks=np.arange(7),xticklabels=labels,ylabel=label);ax.tick_params(axis='x',rotation=35)
    axes[0].legend();f.suptitle('Ring8: one paired C8-stratified bank (4096 base points)')
    f.savefig(outputs[0],dpi=190);plt.close(f)
    f,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for ax,task,title in zip(axes,['gmm_ring8_2d','rotated_product_bimodal_16d'],['Ring8','Product16']):
        for family,label,color in [('state_midpoint4','Fixed midpoint','#0072B2'),('free_stage_midpoint4','Free stages','#D55E00')]:
            r=sorted([r for r in rows if r['dataset']==task and r['family']==family and r['mode']=='full-internal'],key=lambda r:int(r['inner']))
            assert len(r)==2;ax.plot([int(x['inner']) for x in r],[float(x['score']) for x in r],marker='o',label=label,color=color)
        for family,label,color in [('screen_isru_k8','Fixed K8','#888888'),('free_isru','Free ISRU','#009E73')]:
            case=next(c for c in cases if c['dataset']==task and c['family']==family);ax.axhline(case['original_cpu_pooled_sw2'],ls='--',color=color,label=label)
        ax.set(xticks=[4,16],xlabel='Hidden midpoint substeps',ylabel='IID pooled SW2',title=title)
    axes[1].legend(fontsize=8);f.savefig(outputs[1],dpi=190);plt.close(f)
    text=['# Completed numerical and symmetry controls','','All25cases and78scores independently recompute within2.78e-17. The failed','float32 C8bundle is preserved; this completed C8cohort uses float64 uniformly.','','## Ring8 symmetry intervention','','| Model | Original SW2 | Projected SW2 | Original sector L1 |','|---|---:|---:|---:|']
    for label,c in zip(labels,selected):
        plain=next(r for r in c['rows'] if r['mode']=='original');projected=next(r for r in c['rows'] if r['mode']=='projected')
        text.append(f"| {label} | {plain['stratified_sw2']:.3f} | {projected['stratified_sw2']:.3f} | {plain['sector_l1']:.3f} |")
    text+=['','State flow leads ordinary controls before projection; its lead disappears or','reverses afterwards on this paired bank. Free ISRU has the lowest projected','score. Projected sector counts are equal by construction. These observations','support an asymmetry/mode-allocation explanation of this particular Ring8 gap,','not an intrinsic equivariance guarantee of the hidden ODE.','','The8rotations of each point are correlated. There are4096independent base','points, not32768independent observations. No bank-level confidence interval','is estimated from the rotations, and these absolute scores are not mixed with','the iid pooled-bank scores. The released stage-operator model is not included.','','![Symmetry intervention](figures/symmetry-full.png)','','## Internal refinement','','Fixed midpoint changes Ring8 0.071->0.074 and Product16 0.052->0.055 when','refined4->16. Free stages change0.071->0.071 and0.052->0.053. This keeps the','Ring8 within-family signal while shrinking Product16\'s fixed-midpoint gain.','','![Full-sample internal refinement](figures/internal-resolution-full.png)','','The14outer-resolution cases use512common points each. Their largest absolute','SW2 change at Heun128->512 is7.065e-4. This is a bounded numerical observation,','not a full-bank error bound. The new released-stage models were not part of','those14cases and require their own outer-refinement check.']
    (study/'numerical-findings.md').write_text('\n'.join(text)+'\n')
    (study/'numerical-figures.json').write_text(json.dumps(dict(passed=True,source_sha256=sha(__file__),artifacts={str(p.relative_to(study)):sha(p) for p in outputs}),indent=2)+'\n')
    print(json.dumps(dict(passed=True,figures=len(outputs))))

if __name__=='__main__':main()
