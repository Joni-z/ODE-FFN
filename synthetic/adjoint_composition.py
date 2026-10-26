"""Exact degree expansion of the complete residual-network input adjoint."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from synthetic.explicit_adjoint import norm_primal,norm_adjoint,ffn_primal,ffn_adjoint


class AdjointComposition(nn.Module):
    def __init__(self,raw:nn.Module):
        super().__init__()
        if raw.depth!=3 or len(raw.blocks)!=3:
            raise ValueError('this exact diagnostic fixes the native three blocks')
        self.raw=raw
        self.dim=raw.dim

    def components(self,x:torch.Tensor,t:torch.Tensor)->torch.Tensor:
        if x.ndim!=2 or x.shape[1]!=self.dim or t.shape!=(len(x),1):
            raise ValueError('expected native batch states and times')
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
        seed=norm_adjoint(norm_cache,F.linear(x,self.raw.output.weight.T))
        coefficients=[seed]
        for block,cache,scale,ffn_cache in reversed(history):
            next_coefficients=list(coefficients)+[torch.zeros_like(seed)]
            for degree,q in enumerate(coefficients):
                dual=ffn_adjoint(ffn_cache,q*block.residual_scale)
                dual=norm_adjoint(cache,dual*(1+scale))
                next_coefficients[degree+1]=next_coefficients[degree+1]+dual
            coefficients=next_coefficients
        components=[.5*F.linear(q,self.raw.input.weight.T) for q in coefficients]
        components[0]=components[0]+.5*value
        return torch.stack(components,dim=1)

    def forward(self,x,t):
        return self.components(x,t).sum(1)
