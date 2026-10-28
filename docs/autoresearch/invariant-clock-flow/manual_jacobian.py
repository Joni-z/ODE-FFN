"""Explicit complete-network state derivative, independent of network AD."""
import torch
from invariant_clocks import InvariantClockFFN,IdentitySiLU

def silu_prime(x):
    s=torch.sigmoid(x);return s+x*s*(1-s)

def norm_jacobian(norm,x):
    d=len(x);q=x-x.mean();variance=q.square().mean()+norm.eps
    base=torch.eye(d,dtype=x.dtype,device=x.device)-torch.ones(d,d,dtype=x.dtype,device=x.device)/d-torch.outer(q,q)/(d*variance)
    return norm.weight[:,None]*base/torch.sqrt(variance)

def original_layer_jacobian(layer,x):
    assert layer.orders==1 and layer.gate_source=='input' and layer.spectral_cap is None
    first,last=layer.t_func[0],layer.t_func[2];z=first(x);sig=torch.sigmoid(last(torch.nn.functional.silu(z))/layer.tau)
    grad=((last.weight[0]*silu_prime(z))@first.weight)*(layer.scale/layer.tau)*sig[0]*(1-sig[0])
    duration=layer.shift+layer.scale*sig[0];p=layer.operator()
    return torch.eye(len(x),dtype=x.dtype,device=x.device)+duration*p+torch.outer(p@x,grad)

def ffn_jacobian(ffn,x):
    if isinstance(ffn,InvariantClockFFN):
        first=ffn.ode1(x);second=torch.nn.functional.silu(first)
        return (ffn.ode2.complete_jacobian(second)*silu_prime(first)[None,:])@ffn.ode1.complete_jacobian(x)
    if isinstance(ffn,IdentitySiLU):
        eye=torch.eye(len(x),dtype=x.dtype,device=x.device)
        a,b=(eye+.6*ffn.matrix1,eye+.6*ffn.matrix2) if ffn.kind=='fixed06_identity' else (ffn.matrix1,ffn.matrix2)
        return (b*silu_prime(a@x)[None,:])@a
    if hasattr(ffn,'ode1'):
        first=ffn.ode1(x);second=torch.nn.functional.silu(first)
        return (original_layer_jacobian(ffn.ode2,second)*silu_prime(first)[None,:])@original_layer_jacobian(ffn.ode1,x)
    gate,value=ffn.w12(x).chunk(2,-1);a,b=ffn.w12.weight.chunk(2,0)
    return ffn.w3.weight@((value*silu_prime(gate))[:,None]*a+torch.nn.functional.silu(gate)[:,None]*b)

@torch.no_grad()
def network_jacobian(model,x,t):
    raw=model.raw;context=raw.time(t.reshape(1,1))[0];hidden=raw.input(x);jac=raw.input.weight.clone()
    for block in raw.blocks:
        scale,shift=block.condition(context).chunk(2,-1);z=block.norm(hidden)*(1+scale)+shift
        delta=ffn_jacobian(block.ffn,z)@((1+scale)[:,None]*norm_jacobian(block.norm,hidden))
        jac=(torch.eye(len(hidden),dtype=x.dtype,device=x.device)+block.residual_scale*delta)@jac
        hidden=hidden+block.residual_scale*block.ffn(z)
    return raw.output.weight@norm_jacobian(raw.norm,hidden)@jac
