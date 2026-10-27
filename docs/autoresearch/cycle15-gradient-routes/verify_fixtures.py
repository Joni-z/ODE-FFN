"""Independent small dense fixtures, no fitted checkpoint access."""
import json
import torch
import numpy as np
from types import SimpleNamespace
from gradient_core import projector,adam,clip

torch.set_num_threads(1);torch.manual_seed(0);dtype=torch.float64
q,_=np.linalg.qr(np.random.default_rng(0).standard_normal((128,2)))
w=torch.randn(128,64,dtype=dtype);gamma=torch.randn(64,dtype=dtype)
raw=SimpleNamespace(output=SimpleNamespace(weight=w),norm=SimpleNamespace(weight=gamma))
t,rank=projector(SimpleNamespace(raw=raw),q);t=torch.from_numpy(t)
h=torch.randn(7,64,dtype=dtype,requires_grad=True)
z=torch.nn.functional.layer_norm(h,(64,),eps=1e-5)*gamma
y=z@w.T;cot=torch.randn_like(y)@t
hidden=torch.autograd.grad((y*cot).sum(),h)[0]
assert hidden.abs().max()<1e-10 and rank==63
zero=SimpleNamespace(raw=SimpleNamespace(output=SimpleNamespace(weight=w*0),norm=raw.norm))
t0,k=projector(zero,q);assert k==126
np.testing.assert_allclose(t0,np.eye(128)-q@q.T,atol=1e-12)
a=torch.tensor([1.,2.],dtype=dtype);b=-a
assert (a+b).square().sum()==0 and a.square().sum()+b.square().sum()>0
p=torch.nn.Parameter(torch.tensor([.2,-.4,.7],dtype=dtype));opt=torch.optim.AdamW([p],lr=3e-4,weight_decay=0.)
for i in range(4):p.grad=torch.tensor([.1+i*.01,-.2,.03],dtype=dtype);opt.step()
state=opt.state_dict();g=torch.tensor([.1,-2.,.5],dtype=dtype)
d=adam(g,state,[dict(start=0,end=3)],clip(g));old=p.detach().clone()
p.grad=g.clone();actual=torch.nn.utils.clip_grad_norm_([p],1.)
opt.step();torch.testing.assert_close(old-p,d,atol=1e-16,rtol=1e-11)
assert abs(float(actual)-float(g.norm()))<1e-14
print(json.dumps(dict(status='PASS',null_hidden_max=float(hidden.abs().max()),cold_dimension=k,
    fitted_dimension=rank,Adam_update_max=float((old-p-d).detach().abs().max()),signed_cross_terms=True)))
