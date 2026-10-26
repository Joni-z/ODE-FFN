"""Independent manual features and pivoted-QR residual-probe replay."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import torch
import psutil

ARMS=('ode_k2_direct','swiglu128_direct','swiglu192_direct','swiglu384_direct');TIMES=(.1,.3,.5,.7,.9)
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def augment(h):return np.concatenate([np.ones((len(h),1)),h],axis=1)

def qr_fit(x,y):
    singular=np.linalg.svd(x,compute_uv=False);rank=int(np.sum(singular>1e-10*singular[0]))
    q,r,pivot=la.qr(x,mode='economic',pivoting=True);c=np.zeros((x.shape[1],y.shape[1]))
    c[pivot[:rank]]=la.solve_triangular(r[:rank,:rank],q[:,:rank].T@y)
    return c,rank,float(np.linalg.norm(x.T@(x@c-y))/max(np.linalg.norm(x)*np.linalg.norm(y),1e-30))

def apply(h,c,rotation):
    mid=len(h)//2;p=np.concatenate([augment(h[:mid])@c[1],augment(h[mid:])@c[0]])
    return np.stack([p[:,:16]@rotation,p[:,16:]@rotation],axis=1)

def expected_sites(arm):
    common=('z','ffn','residual');extra=('clock1','solve1','increment1','middle','clock2','increment2') if arm=='ode_k2_direct' else ()
    return {'input','post_norm'}|{f'b{i}.{s}' for i in range(3) for s in common+extra}

def main(root):
    torch.set_num_threads(1);started=time.monotonic();out=root/'output';dest=root/'independent-audit.json'
    assert load(out/'status.json')['state']=='COMPLETE' and not dest.exists()
    base=Path('/Users/mr.z/ode-workspace-prep/architecture-search');models=base/'conservative_readout_20261026/output';probes=base/'coordinate_dependence_20261026/output-v2';previous=base/'frozen_features_20261026/output'
    bindings=load(out/'input-bindings.json')
    for p,wanted in bindings.items():assert sha(p)==wanted,p
    sys.path.insert(0,str(root/'source'))
    from manual_features import manual
    import manual_features as manual_module
    assert Path(manual_module.__file__).resolve().is_relative_to(root/'source')
    provenance=load(models/'provenance.json');old=Path(provenance['source_root'])
    for name,wanted in provenance['source_sha256'].items():assert sha(old/name)==wanted
    sys.path.insert(0,str(old))
    from synthetic.conservative_readout import build
    from synthetic.train import restore_model
    pop=load(probes/'population.json');rotation=np.array(pop['rotation']);mu=np.array(pop['locations']);settings=load(models/'study.json')['arms']
    with np.load(probes/'standardized-draws.npz') as d:pair=(d['sign_A']*mu+.2*d['target_noise_A']-d['source_A'])@rotation.T
    banks=[]
    for t in TIMES:
        with np.load(probes/'banks'/f'time-{t:.1f}.npz') as d:banks.append({k:d[k] for k in d.files})
    expected={(r['arm'],r['site'],r['label'],r['time'],r['evaluation_fold']):r for r in load(out/'rows.json')}
    max_train=0.;max_pred=0.;max_metric=0.;max_compat=0.;max_normal=0.;fits=0;checked=0;feature_tensors=0
    for arm in ARMS:
        setting=settings[arm];model=build(16,setting['family'],setting['readout']).eval()
        restore_model(model,torch.load(models/'runs'/'rotated_product_bimodal_16d'/arm/'step_0005000.pt',map_location='cpu',weights_only=False))
        for p in model.parameters():p.requires_grad_(False)
        model=model.double();frozen=[];targets=[];names=expected_sites(arm)
        for t,bank in zip(TIMES,banks):
            f,value=manual(model,bank['A']@rotation.T,t,arm);assert set(f)==names
            with np.load(out/arm/f'features_{t:.1f}.npz') as saved:
                assert set(saved.files)==names
                for site in names:np.testing.assert_array_equal(f[site],saved[site]);feature_tensors+=1
            with np.load(probes/'fields'/arm/f'time-{t:.1f}.npz') as d:np.testing.assert_array_equal(value@rotation,d['native'])
            frozen.append(f);targets.append(np.column_stack([bank['oracle']@rotation.T-value,pair-value]))
        n=len(pair);mid=n//2;records=load(out/arm/'fit-records.json');qr_coeff={}
        with np.load(out/arm/'coefficients.npz') as d:svd_coeff={k:d[k] for k in d.files}
        for site in sorted(names):
            coefficients=[]
            for fold,(a,b) in enumerate([(0,mid),(mid,n)]):
                x=augment(np.concatenate([f[site][a:b] for f in frozen]));y=np.concatenate([target[a:b] for target in targets])
                c,rank,normal=qr_fit(x,y);coefficients.append(c);max_normal=max(max_normal,normal)
                info=next(r for r in records if r['site']==site and r['training_fold']==fold);assert rank==info['rank']
                actual=x@c;wanted=x@svd_coeff[site][fold]
                max_train=max(max_train,float(np.max(np.abs(actual-wanted))))
                np.testing.assert_allclose(actual,wanted,atol=1e-7,rtol=1e-8);assert normal<1e-9;fits+=1
            qr_coeff[site]=np.array(coefficients)
        for ti,(t,bank) in enumerate(zip(TIMES,banks)):
            print(json.dumps(dict(phase='internal QR replay',arm=arm,time=t,pid=psutil.Process().pid,rss_mib=psutil.Process().memory_info().rss/2**20)),flush=True)
            with np.load(probes/'fields'/arm/f'time-{t:.1f}.npz') as d:baseA=d['native'];baseH=d['hybrid_diagonal']
            pa={site:baseA[:,None,:]+apply(frozen[ti][site],qr_coeff[site],rotation) for site in names}
            ph={site:np.empty_like(value) for site,value in pa.items()}
            for coordinate in range(16):
                q=bank['B'].copy();q[:,coordinate]=bank['A'][:,coordinate]
                f,value=manual(model,q@rotation.T,t,arm)
                np.testing.assert_array_equal((value@rotation)[:,coordinate],baseH[:,coordinate])
                for site in names:ph[site][:,:,coordinate]=baseH[:,coordinate,None]+apply(f[site],qr_coeff[site],rotation)[:,:,coordinate]
            with np.load(out/arm/f'predictions_{t:.1f}.npz') as saved:
                for site in names:
                    for suffix,value in [('A',pa[site]),('H',ph[site])]:
                        max_pred=max(max_pred,float(np.max(np.abs(value-saved[site+'_'+suffix]))))
                        np.testing.assert_allclose(value,saved[site+'_'+suffix],atol=1e-7,rtol=1e-8)
            with np.load(previous/arm/f'predictions_{t:.1f}.npz') as wanted:
                for suffix,value in [('A',pa['post_norm']),('H',ph['post_norm'])]:
                    max_compat=max(max_compat,float(np.max(np.abs(value-wanted['post_norm_shared_'+suffix]))))
                    np.testing.assert_allclose(value,wanted['post_norm_shared_'+suffix],atol=1e-7,rtol=1e-8)
            groups=[('native','native',baseA,baseH)]
            for site in names:
                for j,label in enumerate(['mean','pair']):groups.append((site,label,pa[site][:,j],ph[site][:,j]))
            for site,label,a,h in groups:
                e=a-bank['oracle'];g=h-bank['oracle']
                for fold,(first,last) in enumerate([(0,mid),(mid,n)]):
                    values=dict(total=float(((e[first:last]**2+g[first:last]**2)/2).mean()),own=float((e[first:last]*g[first:last]).mean()),other=float(((e[first:last]-g[first:last])**2/2).mean()))
                    row=expected[arm,site,label,t,fold]
                    for key,value in values.items():
                        max_metric=max(max_metric,abs(value-row[key]));np.testing.assert_allclose(value,row[key],atol=1e-9,rtol=1e-8)
                    checked+=1
        del model,frozen,targets,pa,ph
    assert checked==len(expected)==1280 and fits==124 and feature_tensors==310
    for p,wanted in bindings.items():assert sha(p)==wanted,p
    result=dict(status='PASS',qr_fits=fits,feature_tensors_bitwise=feature_tensors,heldout_rows=checked,native_hybrid_coordinate_replays=320,
        max_training_prediction_error=max_train,max_heldout_prediction_error=max_pred,max_metric_error=max_metric,
        max_old_final_probe_error=max_compat,max_relative_normal_residual=max_normal,elapsed_seconds=time.monotonic()-started,
        source_sha256=sha(__file__),manual_source_sha256=sha(root/'source/manual_features.py'),input_bindings_sha256=sha(out/'input-bindings.json'))
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
