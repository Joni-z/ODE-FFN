"""Native LayerNorm values with explicit differentiable backward/JVP formulas.

Used only by the directional-derivative diagnostic: installed native JVP
higher derivatives fail finite differences. No normalization parameter changes.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def normalized(x: torch.Tensor,eps: float) -> tuple[torch.Tensor,torch.Tensor]:
    centered=x-x.mean(-1,keepdim=True)
    inverse=torch.rsqrt(centered.square().mean(-1,keepdim=True)+eps)
    return centered*inverse,inverse


class _LayerNorm(torch.autograd.Function):
    generate_vmap_rule=True

    @staticmethod
    def forward(x,weight,bias,eps):
        return F.layer_norm(x,(x.shape[-1],),weight,bias,eps)

    @staticmethod
    def setup_context(ctx,inputs,output):
        x,weight,bias,eps=inputs
        ctx.save_for_backward(x,weight,bias)
        ctx.save_for_forward(x,weight,bias)
        ctx.eps=eps

    @staticmethod
    def backward(ctx,output_gradient):
        x,weight,bias=ctx.saved_tensors
        z,inverse=normalized(x,ctx.eps)
        g=output_gradient*weight
        dx=inverse*(g-g.mean(-1,keepdim=True)-z*(g*z).mean(-1,keepdim=True))
        dims=tuple(range(x.ndim-1))
        dw=(output_gradient*z).sum(dims) if dims else output_gradient*z
        db=output_gradient.sum(dims) if dims else output_gradient
        return dx,dw,db,None

    @staticmethod
    def jvp(ctx,dx,dw,db,deps):
        x,weight,bias=ctx.saved_tensors
        z,inverse=normalized(x,ctx.eps)
        dc=dx-dx.mean(-1,keepdim=True)
        dz=inverse*(dc-z*(dc*z).mean(-1,keepdim=True))
        result=weight*dz
        if dw is not None:result=result+dw*z
        if db is not None:result=result+db
        return result


class DifferentiableLayerNorm(nn.LayerNorm):
    @classmethod
    def from_native(cls,source: nn.LayerNorm):
        if len(source.normalized_shape)!=1 or source.weight is None or source.bias is None:
            raise ValueError('the synthetic backbone uses affine last-axis LayerNorm')
        result=cls(source.normalized_shape,eps=source.eps,
                   device=source.weight.device,dtype=source.weight.dtype)
        result.weight,result.bias=source.weight,source.bias
        result.train(source.training)
        return result

    def forward(self,x):
        return _LayerNorm.apply(x,self.weight,self.bias,self.eps)


def replace_normalizations(module: nn.Module) -> None:
    for name,child in list(module.named_children()):
        if isinstance(child,nn.LayerNorm):
            setattr(module,name,DifferentiableLayerNorm.from_native(child))
        else:replace_normalizations(child)
