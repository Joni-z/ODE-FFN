"""Independent QR/projected-loss audit of every frozen gradient probe."""
from __future__ import annotations
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
import torch

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args();root=args.root.resolve();out=args.output.resolve();out.mkdir(exist_ok=False)
sys.path.insert(0,str(root/'source'));import study
from gradient_core import layout,groups

torch.set_num_threads(1);start=time.monotonic();study.verify(root)
original={str(p):study.sha(p) for p in (root/'output').rglob('*') if p.is_file()}
s,_,geo,native,core,_=study.setup(root);q=np.load(root/'basis.npy');counts=0;maxdiff=0.;maxproj=0.;summaries=[]
group_records={(r['law'],r['family'],int(r['step']),r['group']):r for r in csv.DictReader((root/'output/gradient-groups.csv').open())}
checkpoint_records={(r['law'],r['family'],int(r['step'])):r for r in csv.DictReader((root/'output/checkpoints.csv').open())}
centerbasis=np.linalg.qr(np.column_stack((np.ones(64),np.eye(64)[:,:63])))[0][:,1:]

def independent_adam(g,st,rows,scale):
    pg=st['param_groups'][0];b1,b2=pg['betas'];answer=[]
    for row,index in zip(rows,pg['params']):
        grad=g[row['start']:row['end']]*scale;o=st['state'].get(index)
        if o is None:k=1;mu=torch.zeros_like(grad);nu=mu.clone()
        else:k=int(o['step'])+1;mu=o['exp_avg'].double().flatten();nu=o['exp_avg_sq'].double().flatten()
        # Different algebraic arrangement from the formal EMA expression.
        mu=mu+(grad-mu)*(1-b1);nu=nu+(grad.square()-nu)*(1-b2)
        answer.append((mu/(nu/(1-b2**k)).sqrt().add(pg['eps']))*(pg['lr']/(1-b1**k)))
    return torch.cat(answer)

for entry in s['laws']:
    law=entry['law'];bank=geo.training_bank(entry,native)
    with np.load(Path(entry['banks'])/'field.npz') as f:field={k:f[k] for k in f.files}
    for family in study.FAMILIES:
        for step in study.STEPS:
            assert time.monotonic()-start<s['phase_ceiling_seconds']
            key=f'{law}_{family}_{step:07d}';meta=study.load(root/'output/probes'/f'{key}.json')
            with np.load(root/'output/probes'/f'{key}.npz') as f:d={k:f[k] for k in f.files}
            model,st,folder,path=study.restored(root,law,family,step)
            assert study.sha(path)==meta['sha256']
            with np.load(folder/f'field_{step:07d}.npz') as f:np.testing.assert_array_equal(study.replay(model,field),f['prediction'])
            rng=torch.Generator();rng.set_state(st['data_rng']);x,t,target,source=native.draw(bank,rng)
            label=core.regression_target(target,source,t)
            for k,value in dict(x=x.numpy(),t=t.numpy(),label=label.numpy()).items():np.testing.assert_array_equal(d[k],value)
            ng=torch.autograd.grad((model(x,t)-label).square().mean(),list(model.parameters()),allow_unused=True)
            ng=torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for p,g in zip(model.parameters(),ng)])
            np.testing.assert_array_equal(ng.numpy(),d['native_gradient'])
            model.double();rows=layout(model);assert rows==meta['layout']
            w=model.raw.output.weight.detach().numpy();gamma=model.raw.norm.weight.detach().numpy()
            a=np.column_stack(((w*gamma)@centerbasis,q))
            # Q spans the column space for active heads. At the exact zero head
            # the only nonzero columns are the last two, so handle it explicitly.
            if np.count_nonzero(w)==0:u=np.linalg.qr(q,mode='complete')[0];rank=2
            else:u=np.linalg.qr(a,mode='complete')[0];rank=65
            forced=u[:,rank:]@u[:,rank:].T
            maxproj=max(maxproj,float(np.max(np.abs(forced-d['forced']))))
            np.testing.assert_allclose(forced,d['forced'],atol=1e-9,rtol=0)
            projectors=[q@q.T,forced,np.eye(128)-q@q.T-forced]
            pred=model(x.double(),t.double());error=pred-label.double();params=list(model.parameters());parts=[]
            for j,p in enumerate(projectors):
                loss=(error@torch.from_numpy(p)).square().mean()
                g=torch.autograd.grad(loss,params,retain_graph=j<2,allow_unused=True)
                parts.append(torch.cat([(v if v is not None else torch.zeros_like(z)).detach().flatten() for z,v in zip(params,g)]))
            parts=torch.stack(parts);actual=parts.sum(0)
            maxdiff=max(maxdiff,float(np.max(np.abs(parts.numpy()-d['parts']))))
            np.testing.assert_allclose(parts.numpy(),d['parts'],atol=1e-10,rtol=1e-8)
            np.testing.assert_allclose(actual.numpy(),d['full_gradient'],atol=1e-10,rtol=1e-8)
            full=torch.from_numpy(d['full_gradient']);removed=full-torch.from_numpy(d['parts'][1])
            cf=min(1.,1/(float(full.square().sum().sqrt())+1e-6))
            cr=min(1.,1/(float(removed.square().sum().sqrt())+1e-6))
            df=independent_adam(full,st['optimizer'],rows,cf);dr=independent_adam(removed,st['optimizer'],rows,cr)
            np.testing.assert_allclose(df.numpy(),d['update_full'],atol=1e-10,rtol=1e-8)
            np.testing.assert_allclose(dr.numpy(),d['update_removed'],atol=1e-10,rtol=1e-8)
            saved=checkpoint_records[(law,family,step)]
            assert int(saved['forced_dimension'])==128-rank
            for name,p in zip(('tangent_loss','forced_loss','remaining_normal_loss'),projectors):
                energy=float((error.detach()@torch.from_numpy(p)).square().mean())
                np.testing.assert_allclose(energy,float(saved[name]),atol=1e-10,rtol=1e-8)
            for group,idx in groups(rows).items():
                g=parts[:,idx];normsq=(g*g).sum();cross=2*sum(torch.dot(g[i],g[j]) for i,j in ((0,1),(0,2),(1,2)))
                np.testing.assert_allclose(float(actual[idx].square().sum()),float(normsq+cross),atol=1e-10,rtol=1e-8)
                saved=group_records[(law,family,step,group)]
                a,z,r=g;f=full[idx]
                n=lambda v:float(v.square().sum().sqrt())
                cos=lambda a,b:float(torch.dot(a,b))/(n(a)*n(b)) if n(a)*n(b)>0 else None
                den=n(a)**2+n(r)**2
                expected=dict(parameters=len(idx),full_norm=n(f),tangent_norm=n(a),forced_norm=n(z),
                    remaining_normal_norm=n(r),tangent_forced_dot=float(torch.dot(a,z)),
                    tangent_remaining_dot=float(torch.dot(a,r)),forced_remaining_dot=float(torch.dot(z,r)),
                    closure_max=float((f-a-z-r).abs().max()) if len(idx) else 0.,
                    tangent_remaining_cosine=cos(a,r),cancellation_ratio=n(a+r)**2/den if den>0 else None,
                    full_clip=cf,forced_removed_clip=cr,
                    update_relative_change=n(df[idx]-dr[idx])/n(df[idx]) if n(df[idx])>0 else None,
                    update_cosine=cos(df[idx],dr[idx]))
                for name,v in expected.items():
                    if v is None:assert saved[name]=='',(group,name)
                    else:np.testing.assert_allclose(v,float(saved[name]),atol=1e-10,rtol=1e-8)
            counts+=1
            if counts%11==0:print(json.dumps(dict(audited=counts,total=88,seconds=time.monotonic()-start)),flush=True)
assert original=={str(p):study.sha(p) for p in (root/'output').rglob('*') if p.is_file()}
study.verify(root)
study.write(out/'audit.json',dict(status='PASS',checkpoints=counts,field_rows=counts*20000,
    gradient_routes=counts*3,max_gradient_difference=maxdiff,max_projector_difference=maxproj,
    formal_files_unchanged=len(original),seconds=time.monotonic()-start))
print(json.dumps(study.load(out/'audit.json')),flush=True)
