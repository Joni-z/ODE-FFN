"""Same SW2 formula applied to assigned means and centered residuals."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze import HERE,sw2,sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',required=True);p.add_argument('--parent-runtime',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();runtime=Path(a.runtime);parent=Path(a.parent_runtime);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    grid=json.loads((HERE/'grid.json').read_text());manifest=json.loads((runtime/'audit/manifest.json').read_text());means=np.array(manifest['datasets']['gmm_ring8_2d']['distribution']['means'])
    rows=[]
    for cell in grid['cells']:
        original=parent/'lenovo/output/professor_validation14_20261104'/cell['dataset']/cell['source_case']/'seed000';records=[json.loads(line) for line in (original/'rows.jsonl').read_text().splitlines()]
        with np.load(original/'banks.npz',allow_pickle=False) as f:target=np.concatenate([f[f'target_{b}'] for b in range(8)]);directions=f['directions'].copy()
        chosen=sorted([r for r in records if r['checkpoint_step']==cell['checkpoint_step'] and r['heun_steps']==128],key=lambda r:r['bank']);assert [r['bank'] for r in chosen]==list(range(8))
        pieces=[]
        for row in chosen:
            file=original/row['generated_file'];assert sha(file)==row['generated_sha256'];pieces.append(np.load(file,allow_pickle=False))
        generated=np.concatenate(pieces)
        center=lambda x:means[np.argmax(x@means.T,axis=1)]
        gs,ts=center(generated),center(target)
        rows.append(dict(family=cell['family'],checkpoint_step=cell['checkpoint_step'],samples=len(generated),native_sw2=sw2(generated,target,directions),assigned_mean_sw2=sw2(gs,ts,directions),centered_residual_sw2=sw2(generated-gs,target-ts,directions)))
        print(rows[-1],flush=True)
    write(out/'shape-and-mass.csv',rows)
    (out/'verification.json').write_text(json.dumps(dict(passed=True,rows=6,new_model_calls=0,new_training_runs=0,source_sha256=sha(__file__),metric='Same SW2 formula; distinct transformed arrays, no additive identity.'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.6),layout='constrained')
    for family,label,color in [('pd_isru_k1','ISRU-D','#0072B2'),('free_isru','Free ISRU','#D55E00'),('swiglu64','Selected Swi64','#009E73')]:
        group=sorted([r for r in rows if r['family']==family],key=lambda r:r['checkpoint_step'])
        for ax,metric,title in zip(axes,['native_sw2','assigned_mean_sw2','centered_residual_sw2'],['Original samples','Assigned component means','Centered residuals']):
            ax.plot([15,20],[r[metric] for r in group],marker='o',color=color,label=label);ax.set(title=title,xlabel='Checkpoint (thousand updates)',ylabel='SW2',xticks=[15,20]);ax.spines[['right','top']].set_visible(False)
    axes[1].legend(fontsize=8);fig.savefig(out/'shape-and-mass.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
