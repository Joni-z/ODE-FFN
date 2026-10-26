"""Independent QR/refitted-bootstrap replay from original saved fields."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import psutil

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def predict(x,y,weights):
    n=len(x);result=np.empty_like(y);coef=[]
    for a,b,c,d in [(0,n//2,n//2,n),(n//2,n,0,n//2)]:
        w=weights[a:b];keep=w>0;scale=np.sqrt(w[keep,None])
        q,r=np.linalg.qr(x[a:b][keep]*scale,mode='reduced')
        beta=np.linalg.solve(r,q.T@(y[a:b][keep]*scale))
        result[c:d]=x[c:d]@beta;coef.append(beta)
    return result,np.array(coef)

def main(root):
    start=time.monotonic();output=root/'output';raw=Path('/Users/mr.z/ode-workspace-prep/architecture-search/coordinate_dependence_20261026/output-v2')
    assert json.loads((output/'status.json').read_text())['state']=='COMPLETE'
    dest=root/'independent-verification.json';assert not dest.exists()
    summary=json.loads((output/'summary.json').read_text());arms=summary['arms'];times=(.1,.3,.5,.7,.9)
    inputs=json.loads((output/'input-bindings.json').read_text())
    for path,wanted in inputs.items():assert digest(path)==wanted,path
    pop=json.loads((raw/'population.json').read_text());mu=np.array(pop['locations']);var=np.array(pop['variances'])
    cases=[];max_pred=0.;max_coef=0.;max_basis=0.;original={k:np.zeros(7) for k in ['total','own','other','affine','full']}
    for t in times:
        with np.load(raw/'banks'/f'time-{t:.1f}.npz') as bank:x=bank['A'];vstar=bank['oracle']
        var_xt=(1-t)**2+t*t*var
        design=np.stack([np.ones_like(x),x/np.sqrt(var_xt+t*t*mu*mu),np.tanh(x*t*mu/var_xt)],axis=-1)
        aa=[];hh=[]
        for arm in arms:
            with np.load(raw/'fields'/arm/f'time-{t:.1f}.npz') as d:
                aa.append(d['native']-vstar);hh.append(d['hybrid_diagonal']-vstar)
        a=np.stack(aa,-1);h=np.stack(hh,-1);mean=(a+h)/2
        T=((a*a+h*h)/2).mean(1);C=(a*h).mean(1);D=((a-h)**2/2).mean(1)
        n,dim,_=a.shape;w=np.ones(n)
        cases.append((design,a,h,T,C,D))
        for k,val in [('total',T),('own',C),('other',D)]:original[k]+=val.mean(0)/5
        with np.load(output/'fits'/f'time-{t:.1f}.npz') as saved:
            max_basis=max(max_basis,float(np.max(np.abs(design-saved['design']))))
            np.testing.assert_allclose(design,saved['design'],atol=1e-15,rtol=1e-15)
            for basis,key in [(2,'affine'),(3,'full')]:
                for i in range(dim):
                    p,co=predict(design[:,i,:basis],mean[:,i],w)
                    max_pred=max(max_pred,float(np.max(np.abs(p-saved[key][:,i]))))
                    max_coef=max(max_coef,float(np.max(np.abs(co-saved[key+'_coefficients'][i]))))
                    np.testing.assert_allclose(p,saved[key][:,i],atol=1e-10,rtol=1e-10)
                    np.testing.assert_allclose(co,saved[key+'_coefficients'][i],atol=1e-9,rtol=1e-10)
                    reduction=(a[:,i]*h[:,i]-(a[:,i]-p)*(h[:,i]-p))
                    original[key]+=reduction.mean(0)/(dim*5)
    for key,value in original.items():np.testing.assert_allclose(value,summary['means'][key],atol=1e-12)
    with np.load(output/'bootstrap.npz') as b:saved_boot={k:b[k] for k in b.files}
    rng=np.random.default_rng(0);replayed={k:[] for k in ['total','affine','full','incremental']}
    for b in range(256):
        w=np.concatenate([rng.multinomial(n//2,np.full(n//2,2/n)) for _ in range(2)])
        np.testing.assert_array_equal(w,saved_boot['weights'][b])
        totals=np.zeros((3,7))
        for design,a,h,T,C,D in cases:
            totals[0]+=(w[:,None]*T).sum(0)/(n*5)
            for basis,pos in [(2,1),(3,2)]:
                for i in range(dim):
                    p,_=predict(design[:,i,:basis],(a[:,i]+h[:,i])/2,w)
                    reduction=a[:,i]*h[:,i]-(a[:,i]-p)*(h[:,i]-p)
                    totals[pos]+=(w[:,None]*reduction).sum(0)/(n*dim*5)
        for key,v in [('total',totals[0]),('affine',totals[1]),('full',totals[2]),('incremental',totals[2]-totals[1])]:replayed[key].append(v)
        if b%16==15:
            print(json.dumps(dict(phase='independent QR bootstrap',completed=b+1,total=256,
                pid=psutil.Process().pid,rss_mib=psutil.Process().memory_info().rss/1024**2)),flush=True)
    replayed={k:np.array(v) for k,v in replayed.items()}
    max_boot=0.
    for key,value in replayed.items():
        max_boot=max(max_boot,float(np.max(np.abs(value-saved_boot[key]))))
        np.testing.assert_allclose(value,saved_boot[key],atol=1e-11,rtol=1e-10)
    checks=[]
    for row in summary['decision']['comparisons']:
        j=arms.index(row['reference']);g=original['total'][0]-original['total'][j]
        gs=replayed['total'][:,0]-replayed['total'][:,j]
        values={}
        for key in ['full','incremental']:
            a=original['full'] if key=='full' else original['full']-original['affine']
            point=(a[0]-a[j])/g;samples=(replayed[key][:,0]-replayed[key][:,j])/gs
            lo,hi=np.quantile(samples,[.025,.975])
            for suffix,wanted in [('share',point),('lower',lo),('upper',hi)]:
                np.testing.assert_allclose(row[key+'_'+suffix],wanted,atol=1e-11)
                values[key+'_'+suffix]=float(wanted)
        checks.append(values)
    support=all(r['full_lower']>=.5 and r['incremental_lower']>=.5 for r in checks)
    reject=all(r['full_upper']<.5 for r in checks)
    label='majority coefficient-response account survives' if support else 'majority coefficient-response account rejected' if reject else 'mixed/inconclusive'
    assert label==summary['decision']['classification']
    for path,wanted in inputs.items():assert digest(path)==wanted,path
    result=dict(status='PASS',cases=35,coordinates_per_case=16,folds=2,refitted_bootstraps=256,
        max_basis_error=max_basis,max_prediction_error=max_pred,max_coefficient_error=max_coef,max_bootstrap_error=max_boot,
        decision=label,elapsed_seconds=time.monotonic()-start,source_sha256=digest(__file__),
        input_bindings_sha256=digest(output/'input-bindings.json'),main_source_sha256=summary['source_sha256'])
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
