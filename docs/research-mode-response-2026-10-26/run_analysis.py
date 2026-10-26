"""Out-of-fold posterior-response diagnosis; no model evaluation or training."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
import psutil

TIMES=(.1,.3,.5,.7,.9)
ARMS=('ode_k2_direct','ode_k2_gradient','swiglu128_direct','swiglu128_gradient',
      'swiglu192_direct','swiglu192_gradient','swiglu384_direct')
REFS=('swiglu128_direct','swiglu192_direct','swiglu384_direct')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def coefficients(x,y,weights=None,method='svd'):
    if weights is not None:
        keep=weights>0
        root=np.sqrt(weights[keep])[:,None]
        x,y=x[keep]*root,y[keep]*root
    if method=='svd':
        value,_,rank,singular=np.linalg.lstsq(x,y,rcond=1e-12)
        if rank!=x.shape[1]:raise ArithmeticError('rank-deficient declared basis')
        return value
    q,r=np.linalg.qr(x,mode='reduced')
    return np.linalg.solve(r,q.T@y)

def crossfit(x,y,weights=None,method='svd'):
    n=len(x);mid=n//2
    if n%2:raise ValueError('even row count required')
    result=np.empty_like(y);cs=[]
    for train,test in ((slice(0,mid),slice(mid,n)),(slice(mid,n),slice(0,mid))):
        w=None if weights is None else weights[train]
        c=coefficients(x[train],y[train],w,method)
        result[test]=x[test]@c;cs.append(c)
    return result,np.stack(cs)

def self_test():
    q=np.array([-2.,-1.,-.25,.25,1.,2.])
    design=np.stack([np.ones(6),q,np.tanh(1.3*q)],axis=1)
    coef=np.array([[.2,-.3],[.4,.6],[-.7,.5]])
    core=design@coef
    z=q**5;z-=design@np.linalg.lstsq(design,z,rcond=None)[0]
    z=z/np.sqrt(np.mean(z*z))
    rows=np.array(list(itertools.product(range(6),[-1,1],[-1,1])))
    idx=rows[:,0];m=core[idx]+z[idx,None]*np.array([.3,-.2])
    ea=m+rows[:,1,None]*np.array([.8,.5]);eh=m+rows[:,2,None]*np.array([.8,.5])
    x=np.tile(design[idx],(2,1));a=np.tile(ea,(2,1));h=np.tile(eh,(2,1));g=(a+h)/2
    pred,c=crossfit(x,g)
    np.testing.assert_allclose(c,np.stack([coef,coef]),atol=1e-12)
    expected=np.tile(core[idx],(2,1))
    np.testing.assert_allclose(pred,expected,atol=1e-12)
    reduction=2*pred*g-pred*pred
    np.testing.assert_allclose(reduction,a*h-(a-pred)*(h-pred),atol=1e-12)
    np.testing.assert_allclose(reduction.mean(0),(core*core).mean(0),atol=1e-12)
    np.testing.assert_allclose(((a-pred)*(h-pred)).mean(0),[.09,.04],atol=1e-12)
    clean=np.tile(core[idx],(2,1))
    fit,_=crossfit(x,clean)
    np.testing.assert_allclose(fit,clean,atol=1e-12)
    changed=clean.copy();changed[:len(clean)//2]+=10
    changed_fit,_=crossfit(x,changed)
    np.testing.assert_allclose(changed_fit[:len(clean)//2],fit[:len(clean)//2],atol=1e-12)
    rng=np.random.default_rng(0);w=rng.multinomial(len(x),np.ones(len(x))/len(x))
    p1,_=crossfit(x,g,w,'svd');p2,_=crossfit(x,g,w,'qr')
    np.testing.assert_allclose(p1,p2,atol=1e-12)
    return {'status':'PASS','checks':['representable response','orthogonal shape residual',
            'paired risk identity','out-of-fold isolation','weighted QR agreement']}

def classify(total,full,incremental,bootstrap):
    rows=[]
    for ref in REFS:
        j=ARMS.index(ref);gap=total[0]-total[j]
        if gap<=0:raise ArithmeticError('nonpositive primary gap')
        ratios=np.array([(full[0]-full[j])/gap,(incremental[0]-incremental[j])/gap])
        bgap=bootstrap['total'][:,0]-bootstrap['total'][:,j]
        if np.any(bgap<=0):raise ArithmeticError('bootstrap gap crossed zero')
        samples=np.stack([(bootstrap['full'][:,0]-bootstrap['full'][:,j])/bgap,
            (bootstrap['incremental'][:,0]-bootstrap['incremental'][:,j])/bgap],axis=1)
        ci=np.quantile(samples,[.025,.975],axis=0)
        rows.append(dict(reference=ref,total_gap=float(gap),full_share=float(ratios[0]),
            incremental_share=float(ratios[1]),full_lower=float(ci[0,0]),full_upper=float(ci[1,0]),
            incremental_lower=float(ci[0,1]),incremental_upper=float(ci[1,1])))
    support=all(r['full_lower']>=.5 and r['incremental_lower']>=.5 for r in rows)
    reject=all(r['full_upper']<.5 for r in rows)
    return dict(classification='majority coefficient-response account survives' if support else
        'majority coefficient-response account rejected' if reject else 'mixed/inconclusive',
        comparisons=rows,architecture_selected=False)

def main(args):
    started=time.monotonic();root=args.input;out=args.output
    out.mkdir(parents=True,exist_ok=False)
    def progress(**record):
        row=dict(elapsed_seconds=time.monotonic()-started,pid=psutil.Process().pid,
            rss_mib=psutil.Process().memory_info().rss/1024**2,**record)
        with (out/'telemetry.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(record),flush=True)
    tests=self_test();inputs={}
    def bind(p):inputs[str(p.resolve())]=sha(p)
    for name in ['population.json','status.json','audit.json']:bind(root/name)
    for clock in TIMES:
        bind(root/'banks'/f'time-{clock:.1f}.npz')
        for arm in ARMS:bind(root/'fields'/arm/f'time-{clock:.1f}.npz')
    (out/'input-bindings.json').write_text(json.dumps(inputs,indent=2)+'\n')
    (out/'source-sha256.json').write_text(json.dumps({str(p.resolve()):sha(p) for p in
        [Path(__file__),Path(__file__).with_name('protocol.md')]},indent=2)+'\n')
    pop=json.loads((root/'population.json').read_text());mu=np.array(pop['locations']);var=np.array(pop['variances'])
    data=[];rows=[];rank_records=[]
    (out/'fits').mkdir()
    for clock in TIMES:
        with np.load(root/'banks'/f'time-{clock:.1f}.npz') as bank:
            a=bank['A'];oracle=bank['oracle']
        d=(1-clock)**2+clock**2*var
        alpha=(clock*var-(1-clock))/d;beta=(1-clock)*mu/d;kappa=clock*mu/d
        nonlinear=np.tanh(a*kappa)
        np.testing.assert_allclose(oracle,alpha*a+beta*nonlinear,atol=1e-13,rtol=1e-13)
        design=np.stack([np.ones_like(a),a/np.sqrt(d+clock**2*mu**2),nonlinear],axis=-1)
        ea=[];eh=[]
        for arm in ARMS:
            with np.load(root/'fields'/arm/f'time-{clock:.1f}.npz') as f:
                u=f['native']-oracle;v=f['hybrid_diagonal']-oracle
                for key,val in [('total',(u*u+v*v)/2),('own',u*v),('other',(u-v)**2/2)]:
                    np.testing.assert_allclose(f[key],val,atol=1e-13,rtol=1e-13)
                ea.append(u);eh.append(v)
        ea=np.stack(ea,-1);eh=np.stack(eh,-1);g=(ea+eh)/2
        n,dim,narms=g.shape;aff=np.empty_like(g);full=np.empty_like(g);ca=[];cf=[]
        for i in range(dim):
            aff[:,i],c1=crossfit(design[:,i,:2],g[:,i]);full[:,i],c2=crossfit(design[:,i],g[:,i])
            ca.append(c1);cf.append(c2)
            for fold,sl in enumerate([slice(0,n//2),slice(n//2,n)]):
                singular=np.linalg.svd(design[sl,i],compute_uv=False)
                rank_records.append(dict(time=clock,coordinate=i,fold=fold,condition=float(singular[0]/singular[-1])))
        total=((ea*ea+eh*eh)/2).mean(1);own=(ea*eh).mean(1);other=((ea-eh)**2/2).mean(1)
        ra=(2*aff*g-aff*aff).mean(1);rf=(2*full*g-full*full).mean(1)
        np.testing.assert_allclose(own-rf,((ea-full)*(eh-full)).mean(1),atol=1e-13)
        np.testing.assert_allclose(total-rf,(((ea-full)**2+(eh-full)**2)/2).mean(1),atol=1e-13)
        np.testing.assert_allclose(((ea-full)-(eh-full))**2/2,(ea-eh)**2/2,atol=1e-13)
        np.savez_compressed(out/'fits'/f'time-{clock:.1f}.npz',design=design,
            affine=aff,full=full,affine_coefficients=np.array(ca),full_coefficients=np.array(cf),
            total=total,own=own,other=other,affine_reduction=ra,full_reduction=rf)
        data.append((design,g,total,own,other))
        rows.append(dict(time=clock,total=total.mean(0).tolist(),own=own.mean(0).tolist(),other=other.mean(0).tolist(),
            affine=ra.mean(0).tolist(),full=rf.mean(0).tolist()))
        progress(phase='original fits',time=clock)
    rng=np.random.default_rng(0);weights=[];boot={k:[] for k in ['total','affine','full','incremental']}
    for b in range(256):
        w=np.concatenate([rng.multinomial(n//2,np.full(n//2,2/n)) for _ in range(2)])
        weights.append(w);bt=np.zeros(narms);ba=np.zeros(narms);bf=np.zeros(narms)
        for design,g,total,own,other in data:
            bt+=np.average(total,axis=0,weights=w)/len(TIMES)
            for i in range(dim):
                a,_=crossfit(design[:,i,:2],g[:,i],w);f,_=crossfit(design[:,i],g[:,i],w)
                ba+=np.average(2*a*g[:,i]-a*a,axis=0,weights=w)/(dim*len(TIMES))
                bf+=np.average(2*f*g[:,i]-f*f,axis=0,weights=w)/(dim*len(TIMES))
        for key,value in [('total',bt),('affine',ba),('full',bf),('incremental',bf-ba)]:boot[key].append(value)
        if b%16==15:progress(phase='refitted bootstrap',completed=b+1,total=256)
    boot={k:np.array(v) for k,v in boot.items()}
    means={key:np.mean([r[key] for r in rows],axis=0) for key in ['total','own','other','affine','full']}
    decision=classify(means['total'],means['full'],means['full']-means['affine'],boot)
    np.savez_compressed(out/'bootstrap.npz',weights=np.array(weights),**boot)
    summary=dict(arms=ARMS,rows=rows,means={k:v.tolist() for k,v in means.items()},decision=decision,
        max_design_condition=max(r['condition'] for r in rank_records),self_tests=tests,
        source_sha256=sha(Path(__file__)),inputs=len(inputs),elapsed_seconds=time.monotonic()-started)
    (out/'condition-numbers.json').write_text(json.dumps(rank_records,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for p,expected in inputs.items():assert sha(p)==expected,p
    (out/'status.json').write_text(json.dumps(dict(state='COMPLETE',decision=decision['classification']))+'\n')
    progress(phase='COMPLETE',decision=decision['classification'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test()))
    else:main(args)
