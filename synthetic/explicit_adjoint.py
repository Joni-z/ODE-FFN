"""Exact ordinary-tensor primal/adjoint sweep for the synthetic readout.

The returned potential-gradient field has no nested input-autograd call.
The optional omitted-clock case is a separate frozen-weight diagnostic function.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from synthetic.models import OriginalODELayer, OriginalODEFFN, SwiGLUFFN


def silu_prime(value:torch.Tensor)->torch.Tensor:
    probability=torch.sigmoid(value)
    return probability*(1+value*(1-probability))


def norm_primal(module:nn.LayerNorm,x:torch.Tensor):
    if module.normalized_shape!=(x.shape[-1],) or not module.elementwise_affine:
        raise ValueError('only the declared affine last-axis LayerNorm is supported')
    centered=x-x.mean(-1,keepdim=True)
    inverse=torch.rsqrt(centered.square().mean(-1,keepdim=True)+module.eps)
    return module(x),(centered*inverse,inverse,module.weight)


def norm_adjoint(cache,q:torch.Tensor)->torch.Tensor:
    normalized,inverse,weight=cache
    value=q*weight
    return inverse*(value-value.mean(-1,keepdim=True)
                    -normalized*(value*normalized).mean(-1,keepdim=True))


def ode_primal(layer:OriginalODELayer,h:torch.Tensor):
    if (type(layer) is not OriginalODELayer or layer.orders!=2 or layer.gate_source!='input'
        or layer.spectral_cap is not None or layer.clock_curvature!=1.
        or layer.gate_transform!='sigmoid' or layer.rate_placement!='after_projection'
        or layer.solve_return!='map' or layer.operator_mode!='multiplicative'
        or layer.forcing_mode!='none'):
        raise ValueError('the explicit identity requires the unchanged scalar K2 solve')
    first,last=layer.t_func[0],layer.t_func[2]
    pre=first(h)
    probability=torch.sigmoid(last(F.silu(pre))/layer.tau)
    duration=probability*layer.scale+layer.shift
    ph=layer.proj(h)
    term=ph*duration
    projected_term=layer.proj(term)
    value=h+term+projected_term*duration/2
    cache=(layer,duration,probability,pre,ph+projected_term)
    return value,cache


def ode_adjoint(cache,q:torch.Tensor,include_clock:bool=True)->torch.Tensor:
    layer,duration,probability,pre,response=cache
    p=layer.proj.weight
    pq=F.linear(q,p.T)
    p2q=F.linear(pq,p.T)
    result=q+duration*pq+.5*duration.square()*p2q
    if include_clock:
        first,last=layer.t_func[0],layer.t_func[2]
        logit_gradient=F.linear(silu_prime(pre)*last.weight,first.weight.T)
        clock_gradient=(layer.scale/layer.tau)*probability*(1-probability)*logit_gradient
        result=result+clock_gradient*(response*q).sum(-1,keepdim=True)
    return result


def ffn_primal(module:nn.Module,h:torch.Tensor):
    if type(module) is OriginalODEFFN:
        if type(module.act) is not nn.SiLU:
            raise ValueError('the original middle SiLU must be retained')
        first,cache1=ode_primal(module.ode1,h)
        second,cache2=ode_primal(module.ode2,F.silu(first))
        return second,('ode',first,cache1,cache2)
    if type(module) is SwiGLUFFN:
        gate,value=module.w12(h).chunk(2,-1)
        return module.w3(F.silu(gate)*value),('swiglu',module,gate,value)
    raise ValueError('unsupported FFN type in the explicit sweep')


def ffn_adjoint(cache,q:torch.Tensor,include_clock:bool=True,include_swiglu_gate:bool=True)->torch.Tensor:
    if cache[0]=='ode':
        _,first,cache1,cache2=cache
        q=ode_adjoint(cache2,q,include_clock)*silu_prime(first)
        return ode_adjoint(cache1,q,include_clock)
    _,module,gate,value=cache
    hidden=F.linear(q,module.w3.weight.T)
    gate_dual=hidden*value*silu_prime(gate) if include_swiglu_gate else torch.zeros_like(hidden)
    dual=torch.cat((gate_dual,hidden*F.silu(gate)),-1)
    return F.linear(dual,module.w12.weight.T)


class ExplicitAdjointNet(nn.Module):
    """.5*(G + J_G^T x) with exact explicit reverse sensitivities."""
    def __init__(self,raw:nn.Module,*,include_clock:bool=True,include_swiglu_gate:bool=True):
        super().__init__()
        if raw.depth!=3 or len(raw.blocks)!=3:
            raise ValueError('preserve the native three-block network')
        self.raw=raw
        self.dim=raw.dim
        self.include_clock=include_clock
        self.include_swiglu_gate=include_swiglu_gate

    def forward(self,x:torch.Tensor,t:torch.Tensor)->torch.Tensor:
        if x.ndim!=2 or x.shape[1]!=self.dim or t.shape!=(len(x),1):
            raise ValueError('expected x[batch,dim] and t[batch,1]')
        context=self.raw.time(t)
        h=self.raw.input(x)
        history=[]
        for block in self.raw.blocks:
            scale,shift=block.condition(context).chunk(2,-1)
            normalized,norm_cache=norm_primal(block.norm,h)
            update,ffn_cache=ffn_primal(block.ffn,normalized*(1+scale)+shift)
            h=h+block.residual_scale*update
            history.append((block,norm_cache,scale,ffn_cache))
        normalized,norm_cache=norm_primal(self.raw.norm,h)
        value=self.raw.output(normalized)
        q=norm_adjoint(norm_cache,F.linear(x,self.raw.output.weight.T))
        for block,cache,scale,ffn_cache in reversed(history):
            dual=ffn_adjoint(ffn_cache,q*block.residual_scale,self.include_clock,self.include_swiglu_gate)
            q=q+norm_adjoint(cache,dual*(1+scale))
        q=F.linear(q,self.raw.input.weight.T)
        return .5*(value+q)
