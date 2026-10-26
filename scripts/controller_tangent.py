"""Exact shared-parameter tangent kernels for the current independent FFNs.

This is a frozen-network diagnostic. It never updates parameters. Each module
output is perturbed separately, so downstream controllers remain native.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def silu_prime(x: Tensor) -> Tensor:
    probability=x.sigmoid()
    return probability*(1+x*(1-probability))


@contextmanager
def replace_output(module: nn.Module,value: Tensor) -> Iterator[None]:
    handle=module.register_forward_hook(lambda _module,_args,_output:value)
    try:yield
    finally:handle.remove()


def downstream_response(model: nn.Module,module: nn.Module,x: Tensor,t: Tensor,
                        native: Tensor,expected: Tensor) -> Tensor:
    """Return dv/d(module output), including later input-dependent gates.

    Batch-summed gradients assume examplewise computations, as in the current
    LayerNorm VelocityNet. They do not cover cross-example-coupled models.
    """
    anchor=native.detach().clone().requires_grad_(True)
    with replace_output(module,anchor):
        prediction=model(x,t)
        torch.testing.assert_close(prediction,expected,atol=2e-12,rtol=2e-12)
        response=torch.stack([torch.autograd.grad(prediction[:,j].sum(),anchor,
            retain_graph=j+1<prediction.shape[1])[0] for j in range(prediction.shape[1])],dim=1)
    return response.detach()


def weighted_response_gram(response: Tensor,feature_gram: Tensor) -> Tensor:
    """K[(i,d),(j,e)] = feature_gram[i,j] dot(response[i,d],response[j,e])."""
    count,dimension,channels=response.shape
    flattened=response.reshape(count*dimension,channels)
    gram=flattened@flattened.T
    gram.reshape(count,dimension,count,dimension).mul_(feature_gram[:,None,:,None])
    return gram


def augmented_gram(features: Tensor) -> Tensor:
    return features@features.T+1


def kernels(model: nn.Module,x: Tensor,t: Tensor) -> dict:
    """Exact Euclidean parameter kernels, keeping the rest of the network fixed.

    The receiving-value RMS equalization is a separate counterfactual feature
    model. Its state-dependent rescaling is not a parameter reparameterization.
    """
    if any(module.training for module in model.modules()):
        raise ValueError('frozen tangent diagnosis requires an eval-mode model')
    ode=list(model.ode_layers())
    modules=ode if ode else [(f'blocks.{i}.ffn',block.ffn) for i,block in enumerate(model.blocks)]
    seen=set()
    for _,module in modules:
        raw_identities=[id(parameter) for _,parameter in module.named_parameters(remove_duplicate=False)]
        identities=set(raw_identities)
        if len(raw_identities)!=len(identities):
            raise ValueError('internal parameter sharing requires the corresponding tangent cross terms')
        if seen & identities:raise ValueError('shared FFN parameters require cross-module tangent terms')
        seen.update(identities)
    captures={};handles=[]
    for name,module in modules:
        def capture(_module,args,out,key=name):
            if not isinstance(out,Tensor):raise ValueError('tensor FFN output required')
            captures[key]=(args[0].detach().clone(),out.detach().clone())
        handles.append(module.register_forward_hook(capture))
    try:
        with torch.no_grad():native=model(x,t).detach()
    finally:
        for handle in handles:handle.remove()
    size=native.numel()
    total=native.new_zeros(size,size)
    controller=total.clone();equalized=total.clone();projection=total.clone()
    responses=[];metadata=[]
    for name,module in modules:
        h,out=captures[name]
        downstream=downstream_response(model,module,x,t,out,native)
        hx=augmented_gram(h)
        if ode:
            if (module.orders!=1 or module.gate_source!='input' or module.spectral_cap is not None
                or module.rate_placement!='after_projection' or module.gate_transform!='sigmoid'):
                raise ValueError('this diagnostic covers independent plain input-sigmoid K1 solves')
            with torch.no_grad():
                pre=module.t_func[0](h);activation=F.silu(pre)
                probability=(module.t_func[-1](activation)/module.tau).sigmoid()
                gate=module.shift+module.scale*probability
                slope=module.scale/module.tau*probability*(1-probability)
                if module.operator_mode=='multiplicative':
                    receiving=F.linear(h,module.proj.weight)
                    p_response=downstream*gate[:,None,:]
                else:
                    receiving=h
                    p_response=downstream*(module.shift+module.scale/2)
                per_channel=downstream*receiving[:,None,:]
                response=per_channel if gate.shape[-1]>1 else per_channel.sum(-1,keepdim=True)
                A=response*slope[:,None,:]
                T=(A@module.t_func[-1].weight)*silu_prime(pre)[:,None,:]
                one=weighted_response_gram(A,augmented_gram(activation))+weighted_response_gram(T,hx)
                scale=receiving.square().mean(-1).sqrt()
                divisor=scale.clamp_min(1e-15)
                equ=one.reshape(len(h),native.shape[1],len(h),native.shape[1])
                equ=equ/divisor[:,None,None,None]/divisor[None,None,:,None]
                controller.add_(one);equalized.add_(equ.reshape(size,size))
                projection.add_(weighted_response_gram(p_response,h@h.T))
                responses.append(response)
                metadata.append(dict(layer=name,receiver_rms_min=float(scale.min()),receiver_rms_mean=float(scale.mean()),
                    receiver_rms_max=float(scale.max()),receiver_zero_fraction=float((scale<1e-15).double().mean()),
                    gate_slope_mean=float(slope.mean()),controller_trace=float(one.trace()),
                    receiver_channels=receiving.shape[-1],gate_channels=gate.shape[-1]))
        else:
            if not hasattr(module,'w12') or not hasattr(module,'w3'):
                raise ValueError('only current SwiGLU reference is supported')
            with torch.no_grad():
                gate,value=module.w12(h).chunk(2,-1)
                active=F.silu(gate);mixed=active*value
                value_response=downstream@module.w3.weight
                gate_response=value_response*(value*silu_prime(gate))[:,None,:]
                linear_response=value_response*active[:,None,:]
                total.add_(weighted_response_gram(gate_response,hx))
                total.add_(weighted_response_gram(linear_response,hx))
                total.add_(weighted_response_gram(downstream,augmented_gram(mixed)))
    if ode:total=controller+projection
    result=dict(prediction=native,components={'full_ffn':total},layers=metadata)
    if ode:
        result['components'].update(controller=controller,controller_value_rms_equalized=equalized,projection_P=projection)
        result['free_gate_response']=torch.cat(responses,-1)
    return result
