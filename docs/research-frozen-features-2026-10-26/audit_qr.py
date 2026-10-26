"""Manual-forward and pivoted-QR replay of every frozen-feature probe."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import torch
import psutil

ARMS=('ode_k2_direct','swiglu128_direct','swiglu192_direct','swiglu384_direct')
TIMES=(.1,.3,.5,.7,.9);SITES=('input','pre_norm','post_norm')

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def augment(h):return np.concatenate((np.ones((len(h),1)),h),axis=1)

def qr_fit(x,y):
    singular=np.linalg.svd(x,compute_uv=False);rank=int(np.sum(singular>1e-10*singular[0]))
    q,r,pivot=la.qr(x,mode='economic',pivoting=True)
    c=np.zeros((x.shape[1],y.shape[1]));c[pivot[:rank]]=la.solve_triangular(r[:rank,:rank],q[:,:rank].T@y)
    normal=float(np.linalg.norm(x.T@(x@c-y))/max(np.linalg.norm(x)*np.linalg.norm(y),1e-30))
    return c,rank,normal

def manual(model,x,t):
    f={site:[] for site in SITES};out=[]
    with torch.no_grad():
        for start in range(0,len(x),256):
            a=torch.from_numpy(np.ascontiguousarray(x[start:start+256]));clock=torch.full((len(a),1),t,dtype=a.dtype)
            context=model.raw.time(clock);h=model.raw.input(a);f['input'].append(h.numpy().copy())
            for block in model.raw.blocks:h=block(h,context)
            f['pre_norm'].append(h.numpy().copy());h=model.raw.norm(h);f['post_norm'].append(h.numpy().copy())
            out.append(model.raw.output(h).numpy())
    return {site:np.concatenate(values) for site,values in f.items()},np.concatenate(out)

def apply(h,c,rotation):
    middle=len(h)//2;physical=[]
    for evaluation,(a,b) in enumerate([(0,middle),(middle,len(h))]):physical.append(augment(h[a:b])@c[1-evaluation])
    p=np.concatenate(physical)
    return np.stack([p[:,:16]@rotation,p[:,16:]@rotation],axis=1)

def main(root):
    torch.set_num_threads(1);started=time.monotonic();output=root/'output'
    assert load(output/'status.json')['state']=='COMPLETE'
    receipt=root/'independent-audit.json';assert not receipt.exists()
    models=Path('/Users/mr.z/ode-workspace-prep/architecture-search/conservative_readout_20261026/output')
    probes=Path('/Users/mr.z/ode-workspace-prep/architecture-search/coordinate_dependence_20261026/output-v2')
    bindings=load(output/'input-bindings.json')
    for name,digest in bindings.items():assert sha(name)==digest
    provenance=load(models/'provenance.json');source=Path(provenance['source_root'])
    for name,digest in provenance['source_sha256'].items():assert sha(source/name)==digest
    sys.path.insert(0,str(source))
    from synthetic.conservative_readout import build
    from synthetic.train import restore_model
    settings=load(models/'study.json')['arms'];rotation=np.array(load(probes/'population.json')['rotation'])
    mu=np.array(load(probes/'population.json')['locations'])
    with np.load(probes/'standardized-draws.npz') as d:pair=(d['sign_A']*mu+.2*d['target_noise_A']-d['source_A'])@rotation.T
    banks=[]
    for t in TIMES:
        with np.load(probes/'banks'/f'time-{t:.1f}.npz') as d:banks.append({k:d[k] for k in d.files})
    y=np.array([np.column_stack((b['oracle']@rotation.T,pair)) for b in banks])
    expected_rows={(r['arm'],r['site'],r['scope'],r['label'],r['time'],r['evaluation_fold']):r for r in load(output/'rows.json')}
    checked=0;max_feature=0.;max_train=0.;max_pred=0.;max_metric=0.;max_normal=0.;fit_count=0
    for arm in ARMS:
        setting=settings[arm];model=build(16,setting['family'],setting['readout']).eval()
        restore_model(model,torch.load(models/'runs'/'rotated_product_bimodal_16d'/arm/'step_0005000.pt',map_location='cpu',weights_only=False))
        for p in model.parameters():p.requires_grad_(False)
        model=model.double();frozen=[]
        for t,b in zip(TIMES,banks):
            f,v=manual(model,b['A']@rotation.T,t)
            with np.load(output/arm/f'features_{t:.1f}.npz') as saved:
                for site in SITES:
                    max_feature=max(max_feature,float(np.max(np.abs(f[site]-saved[site]))))
                    np.testing.assert_array_equal(f[site],saved[site])
                np.testing.assert_array_equal(v@rotation,saved['native'])
            frozen.append(f)
        n=len(y[0]);mid=n//2;qr_coeff={};fitrecords=load(output/arm/'fit-records.json')
        with np.load(output/arm/'coefficients.npz') as s:svd_coeff={k:s[k] for k in s.files}
        for site in SITES:
            for scope in ['shared','separate']:
                cs=[]
                for time_index in ([None] if scope=='shared' else range(5)):
                    folds=[]
                    for fold,(a,b) in enumerate([(0,mid),(mid,n)]):
                        if time_index is None:
                            x=augment(np.concatenate([f[site][a:b] for f in frozen]));yy=np.concatenate([v[a:b] for v in y])
                        else:x=augment(frozen[time_index][site][a:b]);yy=y[time_index,a:b]
                        c,rank,normal=qr_fit(x,yy);folds.append(c);max_normal=max(max_normal,normal)
                        saved=next(z for z in fitrecords if z['site']==site and z['scope']==scope and z['time_index']==time_index and z['training_fold']==fold)
                        assert rank==saved['rank']
                        oldc=svd_coeff[site+'_'+scope][fold] if time_index is None else svd_coeff[site+'_'+scope][time_index,fold]
                        a_pred=x@c;b_pred=x@oldc
                        max_train=max(max_train,float(np.max(np.abs(a_pred-b_pred))))
                        np.testing.assert_allclose(a_pred,b_pred,atol=1e-7,rtol=1e-8)
                        assert normal<1e-9
                        fit_count+=1
                    cs.append(np.array(folds))
                qr_coeff[site+'_'+scope]=cs[0] if scope=='shared' else np.array(cs)
        for index,(t,bank) in enumerate(zip(TIMES,banks)):
            print(json.dumps(dict(phase='QR held-out replay',arm=arm,time=t,pid=psutil.Process().pid,rss_mib=psutil.Process().memory_info().rss/2**20)),flush=True)
            a_values={};h_values={}
            for key,c in qr_coeff.items():
                site,scope=key.rsplit('_',1);cc=c if scope=='shared' else c[index]
                a_values[key]=apply(frozen[index][site],cc,rotation);h_values[key]=np.empty_like(a_values[key])
            with np.load(probes/'fields'/arm/f'time-{t:.1f}.npz') as d:nativeA=d['native'];nativeH=d['hybrid_diagonal']
            for i in range(16):
                q=bank['B'].copy();q[:,i]=bank['A'][:,i]
                f,v=manual(model,q@rotation.T,t);np.testing.assert_array_equal((v@rotation)[:,i],nativeH[:,i])
                for key,c in qr_coeff.items():
                    site,scope=key.rsplit('_',1);cc=c if scope=='shared' else c[index]
                    h_values[key][:,:,i]=apply(f[site],cc,rotation)[:,:,i]
            with np.load(output/arm/f'predictions_{t:.1f}.npz') as saved:
                for key in a_values:
                    for suffix,values in [('A',a_values[key]),('H',h_values[key])]:
                        max_pred=max(max_pred,float(np.max(np.abs(values-saved[key+'_'+suffix]))))
                        np.testing.assert_allclose(values,saved[key+'_'+suffix],atol=1e-7,rtol=1e-8)
            groups=[('native','native','native',nativeA,nativeH)]
            for key in a_values:
                site,scope=key.rsplit('_',1)
                for j,label in enumerate(['mean','pair']):groups.append((site,scope,label,a_values[key][:,j],h_values[key][:,j]))
            for site,scope,label,a,h in groups:
                e=a-bank['oracle'];g=h-bank['oracle']
                for fold,(first,last) in enumerate([(0,mid),(mid,n)]):
                    row=expected_rows[(arm,site,scope,label,t,fold)]
                    actual=dict(total=float(((e[first:last]**2+g[first:last]**2)/2).mean()),
                        own=float((e[first:last]*g[first:last]).mean()),other=float(((e[first:last]-g[first:last])**2/2).mean()))
                    for k,value in actual.items():
                        max_metric=max(max_metric,abs(value-row[k]));np.testing.assert_allclose(value,row[k],atol=1e-9,rtol=1e-8)
                    checked+=1
        del model,frozen
    assert checked==len(expected_rows)==520
    for name,digest in bindings.items():assert sha(name)==digest
    result=dict(status='PASS',manual_feature_cases=20,native_hybrid_coordinate_replays=320,qr_fits=fit_count,
        heldout_metric_rows=checked,max_feature_error=max_feature,max_training_prediction_error=max_train,
        max_heldout_prediction_error=max_pred,max_metric_error=max_metric,max_relative_normal_residual=max_normal,
        elapsed_seconds=time.monotonic()-started,source_sha256=sha(__file__),input_bindings_sha256=sha(output/'input-bindings.json'))
    receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
