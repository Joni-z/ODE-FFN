"""Finite-bank Gaussian conditional moments; no learned prior or hidden model."""
import math
import torch

@torch.no_grad()
def posterior(bank,z,t,*,direct=False,batch=64):
    assert bank.ndim==z.ndim==2 and bank.dtype==z.dtype==torch.float64
    variance=(1-t)**2;norms=bank.square().sum(1);parts=[]
    for q in z.split(batch):
        if direct:
            logits=-torch.cdist(q,t*bank,p=2,compute_mode='donot_use_mm_for_euclid_dist').square()/(2*variance)
            common=0.
        else:
            logits=(t*(q@bank.T)-.5*t*t*norms[None])/variance
            common=q.square().sum(1)/(2*variance)
        logz=torch.logsumexp(logits,1);w=torch.softmax(logits,1);mean=w@bank
        trace=(w@norms-mean.square().sum(1)).clamp_min(0)
        top,index=w.max(1)
        parts.append(dict(mean=mean,variance_trace=trace,effective_support=1/w.square().sum(1),
            entropy=-(w*w.clamp_min(torch.finfo(w.dtype).tiny).log()).sum(1),top_weight=top,top_index=index,
            log_density=logz-common-math.log(len(bank))-.5*bank.shape[1]*math.log(2*math.pi*variance)))
    return {k:torch.cat([v[k] for v in parts]).numpy() for k in parts[0]}

def fixtures():
    g=torch.Generator().manual_seed(0);bank=torch.randn(13,7,generator=g,dtype=torch.float64)
    z=torch.randn(5,7,generator=g,dtype=torch.float64);maximum=0.
    for t in (13/128,64/128,115/128):
        a=posterior(bank,z,t);b=posterior(bank,z,t,direct=True,batch=2)
        for key in a:
            if key=='top_index':assert (a[key]==b[key]).all()
            else:
                torch.testing.assert_close(torch.from_numpy(a[key]),torch.from_numpy(b[key]),rtol=3e-12,atol=3e-12)
                maximum=max(maximum,float(torch.from_numpy(abs(a[key]-b[key])).max()))
        q=z[0].clone().requires_grad_(True);logits=-(q-t*bank).square().sum(1)/(2*(1-t)**2)
        w=torch.softmax(logits,0);mean=w@bank
        automatic=torch.autograd.functional.jacobian(lambda x:torch.softmax(-(x-t*bank).square().sum(1)/(2*(1-t)**2),0)@bank,q)
        centered=bank-mean;cov=centered.T@(w[:,None]*centered)
        torch.testing.assert_close(automatic,t*cov/(1-t)**2,rtol=3e-12,atol=3e-12)
        pred=torch.sin(torch.arange(7,dtype=torch.float64));risk=(w*(bank-pred).square().sum(1)).sum()
        torch.testing.assert_close(risk,(pred-mean).square().sum()+torch.trace(cov),rtol=3e-12,atol=3e-12)
    return dict(status='PASS',times=3,queries=15,direct_likelihood_max_absolute_error=maximum,posterior_derivative_and_risk_identity=True)
