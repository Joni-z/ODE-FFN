"""Fixed professor core and direct ordinary coordinates; data-free witnesses."""
import types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

C=float(np.float32(.6))
class PlainSiLU(nn.Module):
    def __init__(self):
        super().__init__();self.fc1=nn.Linear(64,64,bias=False);self.fc2=nn.Linear(64,64,bias=False)
    def forward(self,x,cond=None,return_aux=False):
        out=self.fc2(F.silu(self.fc1(x)))
        aux={'input_norm':x.detach().norm(dim=-1).mean(),'output_norm':out.detach().norm(dim=-1).mean()}
        return (out,aux) if return_aux else out
def fixed(model,independent=False):
    for block in model.blocks:
        for name in ('ode1','ode2'):
            layer=getattr(block.mlp,name);assert layer.orders==1 and layer.t_from_embed is None
            layer.t_func.requires_grad_(False)
            if independent:
                def forward(self,x,t=None):
                    assert t is None
                    value=F.linear(x,self.proj.weight);base=0+x
                    return base+(value*x.new_tensor(C))/1
            else:
                def forward(self,x,t=None):
                    assert t is None;terms=[x];terms.append(self.proj(terms[-1])*x.new_tensor(C)/1)
                    return sum(terms)
            layer.forward=types.MethodType(forward,layer)
    return model
def direct(model):
    for block in model.blocks:
        parent=block.mlp;p0=parent.ode1.proj.weight
        replacement=PlainSiLU().to(dtype=p0.dtype,device=p0.device)
        with torch.no_grad():
            for j in (1,2):
                p=getattr(parent,f'ode{j}').proj.weight
                getattr(replacement,f'fc{j}').weight.copy_(torch.eye(64,dtype=p.dtype,device=p.device)+C*p)
        block.mlp=replacement
    return model
def fixtures():
    torch.manual_seed(0);dtype=torch.float64;d=5;rate=1e-3;epsilon=1e-8
    p=[nn.Parameter(torch.randn(d,d,dtype=dtype)*.2) for _ in range(2)]
    w=[nn.Parameter(torch.eye(d,dtype=dtype)+C*v.detach()) for v in p]
    theta=nn.Parameter(torch.randn(d,d,dtype=dtype)*.2);other=nn.Parameter(theta.detach().clone())
    opt_p=torch.optim.AdamW([theta,*p],lr=rate,betas=(.9,.99),eps=epsilon,weight_decay=0)
    opt_w=torch.optim.AdamW([dict(params=[other],lr=rate,eps=epsilon),dict(params=w,lr=rate*C,eps=epsilon/C)],betas=(.9,.99),weight_decay=0)
    gradient_error=0.;update_error=0.;forward_error=0.;norm_rows=[]
    for step in range(5):
        x=torch.randn(8,d,dtype=dtype);target=torch.randn(8,d,dtype=dtype)
        opt_p.zero_grad();opt_w.zero_grad();h=x@theta.T;z=h+C*(h@p[0].T);z=F.silu(z);a=z+C*(z@p[1].T)
        b=F.linear(F.silu(F.linear(x@other.T,w[0])),w[1]);forward_error=max(forward_error,float((a-b).detach().abs().max()))
        (a-target).square().mean().backward();(b-target).square().mean().backward()
        for pa,wa in zip(p,w):
            torch.testing.assert_close(pa.grad,C*wa.grad,rtol=1e-11,atol=1e-11)
            gradient_error=max(gradient_error,float((pa.grad-C*wa.grad).abs().max()))
        torch.testing.assert_close(theta.grad,other.grad,rtol=1e-11,atol=1e-11)
        wp_norm=torch.sqrt(other.grad.square().sum()+sum(C*C*v.grad.square().sum() for v in w))
        ww_norm=torch.sqrt(other.grad.square().sum()+sum(v.grad.square().sum() for v in w))
        pre=torch.nn.utils.clip_grad_norm_([theta,*p],.05);factor=(.05/(pre+1e-6)).clamp(max=1.)
        torch.testing.assert_close(pre,wp_norm,rtol=1e-11,atol=1e-11)
        for value in [other,*w]:value.grad.mul_(factor)
        opt_p.step();opt_w.step()
        torch.testing.assert_close(theta,other,rtol=1e-11,atol=1e-11)
        for pa,wa in zip(p,w):
            expected=torch.eye(d,dtype=dtype)+C*pa
            torch.testing.assert_close(wa,expected,rtol=1e-11,atol=1e-11)
            update_error=max(update_error,float((wa-expected).detach().abs().max()))
        norm_rows.append(dict(step=step+1,P_coordinate_norm=float(pre),W_coordinate_norm=float(ww_norm),clip_active=float(pre)>.05))
    assert all(r['clip_active'] for r in norm_rows)
    return dict(status='PASS',gradient_max_error=gradient_error,mapped_Adam_parameter_max_error=update_error,
        forward_max_error=forward_error,steps=5,coordinate_norms=norm_rows,
        transformed_optimizer_is_fixture_only=True)
