"""Summarize actual weights and actual updates, retaining every layer spectrum."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--output',required=True);p.add_argument('--no-plots',action='store_true')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    rows=[];bindings=[]
    for f in sorted(Path(a.runs).rglob('update_spectra.jsonl')):
        cfg=json.loads((f.parent/'config.json').read_text());cell=cfg['cell']
        bindings.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),config_sha256=hashlib.sha256((f.parent/'config.json').read_bytes()).hexdigest()))
        for line in f.read_text().splitlines():
            rec=json.loads(line)
            for name,matrices in rec['matrices'].items():
                for kind in ('weight','delta'):
                    s=np.asarray(matrices[kind]['singular_values'],dtype=np.float64)
                    assert np.isfinite(s).all() and (s>=0).all() and (np.diff(s)<=0).all()
                    energy=float(s@s);prob=s/s.sum() if s.sum() else s;positive=prob[prob>0]
                    stable=float(energy/s[0]**2) if s[0] else 0.
                    entropy=float(np.exp(-(positive*np.log(positive)).sum())) if s.sum() else 0.
                    assert abs(stable-matrices[kind]['stable_rank'])<1e-8
                    assert abs(entropy-matrices[kind]['entropy_rank'])<1e-8
                    rows.append(dict(cell=cell['id'],dataset=cell['dataset'],family=cell['family'],optimizer=cell['optimizer'],lr=cell['projection_lr'],step=rec['step'],matrix=name,kind=kind,stable_rank=stable,entropy_rank=entropy,sigma_max=float(s[0]),sigma_min=float(s[-1]),condition_number=float(s[0]/s[-1]) if s[-1]>0 else None,singular=bool(s[-1]==0),singular_values=s.tolist()))
    assert rows,'No raw spectra found'
    (out/'layers.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    with (out/'layers.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=[k for k in rows[0] if k!='singular_values']);writer.writeheader()
        writer.writerows({k:v for k,v in row.items() if k!='singular_values'} for row in rows)
    endpoint=[]
    keys=sorted({(r['cell'],r['kind']) for r in rows})
    for cell,kind in keys:
        sub=[r for r in rows if r['cell']==cell and r['kind']==kind and r['step']==20000]
        if not sub:continue
        e={k:sub[0][k] for k in ('cell','dataset','family','optimizer','lr','kind')};e['matrices']=len(sub)
        for metric in ('stable_rank','entropy_rank','sigma_max','sigma_min','condition_number'):
            values=[r[metric] for r in sub if r[metric] is not None]
            e[metric+'_mean']=float(np.mean(values)) if values else None
            e[metric+'_median']=float(np.median(values)) if values else None
        endpoint.append(e)
    (out/'endpoints.json').write_text(json.dumps(endpoint,indent=2,allow_nan=False)+'\n')
    (out/'binding.json').write_text(json.dumps(dict(files=bindings,rows=len(rows),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),condition_policy='sigma_max/sigma_min; singular values are not clamped; null means singular'),indent=2)+'\n')
    if a.no_plots:
        print(json.dumps(dict(rows=len(rows),files=len(bindings),output=str(out),plots=False)));return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for dataset in sorted({r['dataset'] for r in rows}):
        fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
        for ax,(kind,metric) in zip(axes.flat,[(k,m) for k in ('weight','delta') for m in ('stable_rank','entropy_rank','condition_number')]):
            for cell in sorted({r['cell'] for r in rows if r['dataset']==dataset}):
                sub=[r for r in rows if r['cell']==cell and r['kind']==kind]
                if sub[0]['family'] not in ('pd_isru_k1','free_isru'):continue
                steps=sorted({r['step'] for r in sub});means=[]
                for t in steps:
                    val=[r[metric] for r in sub if r['step']==t and r[metric] is not None]
                    means.append(np.mean(val) if val else np.nan)
                ax.plot(steps,means,label=sub[0]['family']+'/'+sub[0]['optimizer'])
            if metric=='condition_number':ax.set_yscale('log')
            ax.set(title=kind+' '+metric,xlabel='Update');ax.spines[['top','right']].set_visible(False)
        axes[0,0].legend(fontsize=7);fig.savefig(out/(dataset+'-spectra.png'),dpi=160);plt.close(fig)
    print(json.dumps(dict(rows=len(rows),files=len(bindings),output=str(out))))

if __name__=='__main__':main()
