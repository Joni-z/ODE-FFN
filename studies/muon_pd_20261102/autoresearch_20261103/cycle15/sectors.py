"""Saved-path sector accounting and exact-law component moment diagnostics."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from analyze import HERE,STUDY,TAG,write,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',required=True);p.add_argument('--parent-runtime',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();runtime=Path(a.runtime);parent=Path(a.parent_runtime);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('transport_sector_native_data',STUDY/'reference/synthetic/data.py')
    data=importlib.util.module_from_spec(spec);sys.modules[spec.name]=data;spec.loader.exec_module(data)
    manifest=json.loads((runtime/'audit/manifest.json').read_text());law=data.GaussianMixture(**manifest['datasets']['gmm_ring8_2d']['distribution'])
    grid=json.loads((HERE/'grid.json').read_text());K=len(law.weights);assert K==8
    assert np.allclose(law.weights,np.full(K,1/K),atol=1e-14,rtol=1e-14)
    assert np.allclose(law.variances,np.full_like(law.variances,law.variances[0,0]),atol=1e-14,rtol=1e-14)
    radius=np.linalg.norm(law.means,axis=1);assert np.allclose(radius,radius[0],rtol=1e-14,atol=1e-14)
    theta=np.sort(np.mod(np.arctan2(law.means[:,1],law.means[:,0]),2*np.pi));assert np.allclose(np.diff(np.r_[theta,theta[0]+2*np.pi]),2*np.pi/K,atol=1e-12)
    # At each reflection-fixed boundary ray, exact oracle normal velocity is zero.
    normal_errors=[]
    for angle in theta+np.pi/K:
        tangent=np.array([np.cos(angle),np.sin(angle)]);normal=np.array([-tangent[1],tangent[0]])
        x=np.linspace(.01,6,64)[:,None]*tangent
        for time in [0,.125,.5,.875,1]:normal_errors.append(float(np.max(np.abs(law.velocity(x,np.full((len(x),1),time))@normal))))
    assert max(normal_errors)<1e-10
    sector=lambda x:np.argmax(x@law.means.T,axis=-1)
    with np.load(runtime/'common/common.npz',allow_pickle=False) as f:oracle=f['oracle_path'].astype(float);target=f['target'].copy()
    oracle_labels=sector(oracle);oracle_changes=int(np.sum(oracle_labels[1:]!=oracle_labels[:-1]));assert oracle_changes==0
    events=[];masses=[];endpoint=[];moments=[]
    def path_rows(label,path):
        labels=sector(path);counts=np.stack([np.bincount(y,minlength=K) for y in labels]);freq=counts/labels.shape[1]
        for i in range(len(labels)):
            for k in range(K):masses.append(dict(label=label,step=i,sector=k,count=int(counts[i,k]),frequency=float(freq[i,k])))
            if i:
                events.append(dict(label=label,step=i,time=i/128,gross_crossings=int(np.sum(labels[i]!=labels[i-1])),net_l1_mass_change=float(np.abs(freq[i]-freq[i-1]).sum()),mass_l1_to_analytic=float(np.abs(freq[i]-law.weights).sum())))
        return labels
    path_rows('oracle',oracle)
    def component_moments(label,values):
        labels=sector(values)
        for k in range(K):
            x=values[labels==k].astype(float);assert len(x)>1
            mean=x.mean(axis=0);res=x-mean;cov=res.T@res/len(x)
            bias=float(np.mean((mean-law.means[k])**2));variance=float(np.trace(cov)/2);center=float(np.mean((x-law.means[k])**2))
            assert abs(center-bias-variance)<1e-12
            moments.append(dict(label=label,sector=k,count=len(x),centroid_coordinate_mse=bias,variance_trace_per_coordinate=variance,center_coordinate_mse=center,covariance_frobenius_error=float(np.linalg.norm(cov-np.diag(law.variances[k])))))
    component_moments('bank0_target',target);component_moments('bank0_oracle',oracle[-1])
    pooled_sources=None
    for cell in grid['cells']:
        folder=runtime/'output'/TAG/cell['dataset']/cell['id']/'seed000';path=np.load(folder/'learned_path.npy',allow_pickle=False).astype(float);labels=path_rows(cell['id'],path)
        e2=np.mean((path[-1]-oracle[-1])**2,axis=1);same=labels[-1]==oracle_labels[-1]
        endpoint.append(dict(cell=cell['id'],family=cell['family'],checkpoint_step=cell['checkpoint_step'],oracle_sector_mismatch_fraction=float((~same).mean()),same_sector_error_contribution=float(e2[same].sum()/len(e2)),different_sector_error_contribution=float(e2[~same].sum()/len(e2)),different_sector_energy_fraction=float(e2[~same].sum()/e2.sum()),gross_crossings=int(sum(e['gross_crossings'] for e in events if e['label']==cell['id'])),crossings_after75=int(sum(e['gross_crossings'] for e in events if e['label']==cell['id'] and e['step']>96))))
        original=parent/'lenovo/output/professor_validation14_20261104'/cell['dataset']/cell['source_case']/'seed000';records=[json.loads(line) for line in (original/'rows.jsonl').read_text().splitlines()]
        chosen=sorted([r for r in records if r['checkpoint_step']==cell['checkpoint_step'] and r['heun_steps']==128],key=lambda r:r['bank'])
        values=[]
        for row in chosen:
            file=original/row['generated_file'];assert sha(file)==row['generated_sha256'];values.append(np.load(file,allow_pickle=False))
        component_moments(f"pool_{cell['family']}_{cell['checkpoint_step']}",np.concatenate(values))
        if pooled_sources is None:
            with np.load(original/'banks.npz',allow_pickle=False) as f:
                pooled_sources=np.concatenate([f[f'source_{i}'] for i in range(8)]);pooled_target=np.concatenate([f[f'target_{i}'] for i in range(8)])
    component_moments('pool_target',pooled_target)
    initial=np.bincount(sector(pooled_sources),minlength=K)/len(pooled_sources)
    write(out/'sector-masses.csv',masses);write(out/'sector-events.csv',events);write(out/'endpoint-sector-error.csv',endpoint);write(out/'component-moments.csv',moments)
    receipt=dict(passed=True,new_model_calls=0,new_training_runs=0,oracle_normal_velocity_max=max(normal_errors),oracle_sector_crossings=oracle_changes,pooled_source_sector_frequencies=initial.tolist(),pooled_source_l1_to_uniform=float(np.abs(initial-law.weights).sum()),analytic_component_variance=float(law.variances[0,0]),endpoint_rows=len(endpoint),event_rows=len(events),component_rows=len(moments),source_sha256=sha(__file__),manifest_sha256=sha(runtime/'audit/manifest.json'))
    (out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.6),layout='constrained')
    for ax,(family,title) in zip(axes,[('pd_isru_k1','ISRU-D'),('free_isru','Free ISRU'),('swiglu64','Selected Swi64')]):
        for step,color in [(15000,'#0072B2'),(20000,'#D55E00')]:
            cell=next(c for c in grid['cells'] if c['family']==family and c['checkpoint_step']==step)
            rows=[r for r in events if r['label']==cell['id']];ax.plot([r['time'] for r in rows],np.cumsum([r['gross_crossings'] for r in rows]),color=color,label=f'{step//1000}k')
        ax.axvline(.75,color='black',linestyle=':',linewidth=.8);ax.set(title=title,xlabel='Global FM time',ylabel='Cumulative gross sector crossings');ax.legend();ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'sector-crossings.png',dpi=180);plt.close(fig)
    print(json.dumps(receipt))


if __name__=='__main__':main()
