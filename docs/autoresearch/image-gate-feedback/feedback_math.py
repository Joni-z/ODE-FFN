"""Real-valued native derivatives; no forward intervention or quality metric."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def linear(x,module):
    return F.linear(x,module.weight.detach().double(),None if module.bias is None else module.bias.detach().double())
def silu_prime(x):
    p=torch.sigmoid(x);return p*(1+x*(1-p))
def duration(layer,x):
    return (layer.shift+layer.scale*torch.sigmoid(layer.t_func(x)/layer.tau)).squeeze(-1)
def literal_layer(layer,x,value=None):
    s=(duration(layer,x) if value is None else value).unsqueeze(-1);term=x;out=x
    for j in range(1,layer.orders+1):term=layer.proj(term)*s/j;out=out+term
    return out
def literal_core(module,x,values=None):
    if hasattr(module,'ode1'):
        y=literal_layer(module.ode1,x,None if values is None else values[...,0]);z=F.silu(y)
        return literal_layer(module.ode2,z,None if values is None else values[...,1])
    if hasattr(module,'w12'):
        a,b=module.w12(x).chunk(2,dim=-1);return module.w3(F.silu(a)*b)
    return module.fc2(F.silu(module.fc1(x)))
def anchor_values(module,x):
    c1=duration(module.ode1,x);z=F.silu(literal_layer(module.ode1,x,c1));c2=duration(module.ode2,z)
    return torch.stack((c1,c2),dim=-1)

def local(layer,x):
    assert x.ndim==2 and layer.orders in (1,2) and layer.t_from_embed is None
    p=layer.proj.weight.detach().double();network=layer.t_func
    hidden=linear(x,network[0]);a=F.silu(hidden);logits=linear(a,network[2]).squeeze(-1)/layer.tau
    sigmoid=torch.sigmoid(logits);s=layer.shift+layer.scale*sigmoid
    w1=network[0].weight.detach().double();w2=network[2].weight.detach().double()[0]
    g=(layer.scale/layer.tau*sigmoid*(1-sigmoid))[:,None]*((silu_prime(hidden)*w2)@w1)
    first=x@p.T;v=first;base=torch.eye(len(p),dtype=x.dtype)[None]+s[:,None,None]*p[None]
    y=x+s[:,None]*first
    if layer.orders==2:
        pp=p@p;second=x@pp.T;y=y+.5*s[:,None]**2*second
        base=base+.5*s[:,None,None]**2*pp[None];v=v+s[:,None]*second
    jacobian=base+v[:,:,None]*g[:,None,:]
    return dict(y=y,J=jacobian,R=base,s=s,g=g,v=v,P=p)
def core(module,x):
    if hasattr(module,'ode1'):
        one=local(module.ode1,x);a=silu_prime(one['y']);two=local(module.ode2,F.silu(one['y']))
        return dict(y=two['y'],J=two['J']@(a[:,:,None]*one['J']),R=two['R']@(a[:,:,None]*one['R']))
    if hasattr(module,'w12'):
        a,b=linear(x,module.w12).chunk(2,dim=-1);wa,wb=module.w12.weight.detach().double().chunk(2,dim=0)
        w3=module.w3.weight.detach().double()
        jacobian=(w3[None]*(b*silu_prime(a))[:,None,:])@wa+(w3[None]*F.silu(a)[:,None,:])@wb
        return dict(y=linear(F.silu(a)*b,module.w3),J=jacobian)
    h=linear(x,module.fc1);w1=module.fc1.weight.detach().double();w2=module.fc2.weight.detach().double()
    return dict(y=linear(F.silu(h),module.fc2),J=w2[None]@(silu_prime(h)[:,:,None]*w1))
def spectra(result):
    out=dict(sv_full=torch.linalg.svdvals(result['J']).numpy())
    if 'R' in result:
        out['sv_ref']=torch.linalg.svdvals(result['R']).numpy()
        if 'g' in result:
            out.update(s=result['s'].numpy(),gradient_norm=result['g'].norm(dim=-1).numpy(),
                feedback_norm=(result['v'].norm(dim=-1)*result['g'].norm(dim=-1)).numpy(),
                P_sigma_max=np.array(float(torch.linalg.svdvals(result['P'])[0])))
            scale=out['sv_ref'][:,0];valid=out['sv_ref'][:,-1]>np.finfo(np.float64).eps*len(result['P'])*scale
            chi=np.full(len(scale),np.nan)
            if valid.any():
                q=torch.linalg.solve(result['R'][valid],result['v'][valid,:,None]).squeeze(-1)
                chi[valid]=torch.sum(result['g'][valid]*q,dim=-1).numpy()
            out.update(base_inverse_valid=valid,chi=chi)
        else:
            out['sv_feedback']=torch.linalg.svdvals(result['J']-result['R']).numpy()
            assert np.all(out['sv_feedback'][:,2]<=1e-10+1e-8*out['sv_feedback'][:,0]),'Native two-clock rank bound failed'
    return out

def variance_parts(values):
    x=np.asarray(values,dtype=np.float64);assert x.ndim==3
    grand=x.mean();time_mean=x.mean(axis=(1,2));image_mean=x.mean(axis=2)
    parts=dict(variance_total=float(np.mean((x-grand)**2)),
        variance_time=float(np.mean((time_mean-grand)**2)),
        variance_image=float(np.mean((image_mean-time_mean[:,None])**2)),
        variance_token=float(np.mean((x-image_mean[:,:,None])**2)))
    residual=parts['variance_total']-sum(parts[k] for k in ('variance_time','variance_image','variance_token'))
    assert abs(residual)<1e-12;parts['identity_error']=residual
    for key in ('time','image','token'):parts['fraction_'+key]=parts['variance_'+key]/parts['variance_total'] if parts['variance_total'] else 0.
    return parts

def fixtures():
    torch.manual_seed(0);rows=[]
    class Layer(nn.Module):
        def __init__(self,d,order,hidden=7):
            super().__init__();self.proj=nn.Linear(d,d,bias=False);self.t_func=nn.Sequential(nn.Linear(d,hidden),nn.SiLU(),nn.Linear(hidden,1))
            self.tau=10.;self.scale=.8;self.shift=.2;self.orders=order;self.t_from_embed=None
    class Core(nn.Module):
        def __init__(self,order):
            super().__init__();self.ode1=Layer(5,order);self.ode2=Layer(5,order)
    for order in (1,2):
        module=Core(order).double().requires_grad_(False);x=torch.randn(4,5,dtype=torch.float64)
        actual=core(module,x);observed=torch.func.vmap(torch.func.jacrev(lambda h:literal_core(module,h)))(x)
        torch.testing.assert_close(actual['J'],observed,rtol=1e-10,atol=1e-10)
        values=anchor_values(module,x).detach()
        held=torch.func.vmap(torch.func.jacrev(lambda h,c:literal_core(module,h,c),argnums=0))(x,values)
        torch.testing.assert_close(actual['R'],held,rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(literal_core(module,x,values),actual['y'],rtol=1e-10,atol=1e-10)
        singular=torch.linalg.svdvals(actual['J']-actual['R']);assert singular[:,2].max()<1e-10
        layer=module.ode1;p=layer.proj.weight.detach();e=torch.randn_like(p)*.2;s=duration(layer,x)[:,None]
        def by_p(weight):
            term=x;out=x
            for j in range(1,order+1):term=(term@weight.T)*s/j;out=out+term
            return out
        _,ad=torch.autograd.functional.jvp(by_p,(p,),(e,))
        exact=s*(x@e.T)
        if order==2:exact=exact+.5*s*s*((x@e.T)@p.T+(x@p.T)@e.T)
        torch.testing.assert_close(ad,exact,rtol=1e-10,atol=1e-10)
        rows.append(dict(order=order,core_AD_max_error=float((actual['J']-observed).abs().max()),rank3_residual=float(singular[:,2].max()),P_JVP_max_error=float((ad-exact).abs().max())))
    layer=Layer(3,1,hidden=2).double().requires_grad_(False)
    with torch.no_grad():
        layer.proj.weight.copy_(-.5*torch.eye(3));layer.t_func[0].weight.zero_();layer.t_func[0].bias.zero_()
        layer.t_func[0].weight[0,0]=1;layer.t_func[0].weight[1,0]=-1
        layer.t_func[2].weight.copy_(torch.tensor([[40.,-40.]]));layer.t_func[2].bias.fill_(-200.)
    x=torch.tensor([[5.,0.,0.]],dtype=torch.float64);r=local(layer,x)
    ad=torch.func.vmap(torch.func.jacrev(lambda h:literal_layer(layer,h)))(x)
    torch.testing.assert_close(r['J'],ad,rtol=1e-10,atol=1e-10)
    assert abs(float(torch.linalg.svdvals(ad)[0,0])-1.3)<1e-10
    # Native Heun chain rule on a data-free smooth denoiser.
    a=torch.randn(3,3,dtype=torch.float64)*.1;b=torch.randn(3,3,dtype=torch.float64)*.2;c=torch.randn(3,dtype=torch.float64)
    dt=1/128;heun=[]
    def denoiser(z,t):return a@z+.2*F.silu(b@z)+t*c
    def velocity(z,t):return (denoiser(z,t)-z)/max(1-t,.05)
    def jv(z,t):return (a+.2*(silu_prime(b@z)[:,None]*b)-torch.eye(3,dtype=torch.float64))/max(1-t,.05)
    for j in (13,64,115):
        t=j/128;z=torch.randn(3,dtype=torch.float64);trial=z+dt*velocity(z,t)
        expected=torch.eye(3,dtype=torch.float64)+dt/2*(jv(z,t)+jv(trial,t+dt)@(torch.eye(3,dtype=torch.float64)+dt*jv(z,t)))
        observed=torch.autograd.functional.jacobian(lambda q:q+dt/2*(velocity(q,t)+velocity(q+dt*velocity(q,t),t+dt)),z)
        torch.testing.assert_close(expected,observed,rtol=1e-10,atol=1e-10);heun.append(float((expected-observed).abs().max()))
    variance_parts(np.sin(np.arange(60)).reshape(3,4,5))
    return dict(status='PASS',polynomial_rows=rows,counterexample_P_norm=.5,counterexample_full_gain=1.3,Heun_max_error=max(heun),variance_identity=True)
