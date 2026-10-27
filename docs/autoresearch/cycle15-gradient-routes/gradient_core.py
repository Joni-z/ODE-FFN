"""Frozen output cotangent routes, not a training objective."""
from __future__ import annotations
import numpy as np
import torch


def projector(model, q):
    w=model.raw.output.weight.detach().double().numpy()
    gamma=model.raw.norm.weight.detach().double().numpy()
    center=np.eye(w.shape[1])-np.ones((w.shape[1],w.shape[1]))/w.shape[1]
    b=(w*gamma)@center
    u,s,_=np.linalg.svd(np.column_stack((b,q)),full_matrices=True)
    rank=int(np.sum(s>1e-10*max(float(s.max()),1.)))
    null=u[:,rank:];return null@null.T,128-rank


def routes(model,x,t,label,q,forced):
    params=list(model.parameters());pred=model(x,t);error=(pred-label).detach()
    a=q@q.T;r=torch.eye(128,dtype=x.dtype)-a-forced
    components=[error@a,error@forced,error@r]
    loss=(pred-label).square().mean()
    full=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    values=[]
    for j,e in enumerate(components):
        grads=torch.autograd.grad((pred*e).sum()*2/pred.numel(),params,
            retain_graph=j<2,allow_unused=True)
        values.append(torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten()
            for p,g in zip(params,grads)]))
    full=torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten()
        for p,g in zip(params,full)])
    return full,torch.stack(values),[float(e.square().mean()) for e in components]


def layout(model):
    rows=[];offset=0
    for name,p in model.named_parameters():
        rows.append(dict(name=name,start=offset,end=offset+p.numel(),shape=list(p.shape)))
        offset+=p.numel()
    return rows


def groups(rows):
    masks={key:[] for key in ('all','hidden','readout','P','ffn')}
    for r in rows:
        n=r['name'];readout=n.startswith(('raw.output.','raw.norm.'))
        for key,select in dict(all=True,hidden=not readout,readout=readout,
                P=n.endswith('.proj.weight') and '.ffn.' in n,ffn='.ffn.' in n).items():
            if select:masks[key].extend(range(r['start'],r['end']))
    return {k:torch.tensor(v,dtype=torch.long) for k,v in masks.items()}


def clip(g):return min(1.,1./(float(torch.linalg.vector_norm(g))+1e-6))


def adam(g,state,rows,factor):
    pg=state['param_groups'];assert len(pg)==1;pg=pg[0]
    assert pg['weight_decay']==0 and not pg.get('amsgrad',False) and not pg.get('maximize',False)
    ids=pg['params'];assert len(ids)==len(rows)
    b1,b2=pg['betas'];values=[]
    for r,index in zip(rows,ids):
        grad=g[r['start']:r['end']]*factor;old=state['state'].get(index)
        if old is None:m=torch.zeros_like(grad);v=torch.zeros_like(grad);k=0
        else:m=old['exp_avg'].double().flatten();v=old['exp_avg_sq'].double().flatten();k=int(old['step'])
        m=b1*m+(1-b1)*grad;v=b2*v+(1-b2)*grad.square()
        values.append(pg['lr']*(m/(1-b1**(k+1)))/(torch.sqrt(v/(1-b2**(k+1)))+pg['eps']))
    return torch.cat(values)


def norm(g):return float(torch.linalg.vector_norm(g))
def cosine(a,b):
    den=norm(a)*norm(b);return float(torch.dot(a,b))/den if den>0 else None


def summarize(full,parts,rows,state):
    cf=clip(full);removed=full-parts[1];cr=clip(removed)
    df=adam(full,state,rows,cf);dr=adam(removed,state,rows,cr)
    result=[]
    for name,idx in groups(rows).items():
        f=full[idx];a,z,r=parts[:,idx];den=norm(a)**2+norm(r)**2
        entry=dict(group=name,parameters=len(idx),full_norm=norm(f),tangent_norm=norm(a),
            forced_norm=norm(z),remaining_normal_norm=norm(r),
            tangent_forced_dot=float(torch.dot(a,z)),tangent_remaining_dot=float(torch.dot(a,r)),
            forced_remaining_dot=float(torch.dot(z,r)),
            closure_max=float((f-a-z-r).abs().max()) if len(idx) else 0.,
            tangent_remaining_cosine=cosine(a,r),
            cancellation_ratio=norm(a+r)**2/den if den>0 else None,
            full_clip=cf,forced_removed_clip=cr,
            update_relative_change=norm(df[idx]-dr[idx])/norm(df[idx]) if norm(df[idx])>0 else None,
            update_cosine=cosine(df[idx],dr[idx]))
        result.append(entry)
    return result,df,dr
