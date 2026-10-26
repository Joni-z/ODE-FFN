"""Parameter-free decomposition of pre-LayerNorm residual updates."""
from __future__ import annotations

import torch
from torch import nn


def decompose(hidden: torch.Tensor, branch: torch.Tensor):
    if hidden.ndim!=2 or hidden.shape!=branch.shape:
        raise ValueError("matching[batch,width] states required")
    centered=hidden-hidden.mean(-1,keepdim=True)
    uniform=branch.mean(-1,keepdim=True).expand_as(branch)
    squared_radius=centered.square().sum(-1,keepdim=True)
    safe=torch.where(squared_radius>0,squared_radius,torch.ones_like(squared_radius))
    beta=((branch-uniform)*centered).sum(-1,keepdim=True)/safe
    radial=beta*centered
    tangent=branch-uniform-radial
    return centered,uniform,radial,tangent,beta


class GeometryNet(nn.Module):
    def __init__(self,parent,variant: str="native",block_index: int | None=None):
        super().__init__()
        if variant not in ("native","drop_uniform","drop_radial","drop_tangent"):
            raise ValueError("unknown geometry intervention")
        if block_index not in (None,0,1,2):
            raise ValueError("unsupported block index")
        self.raw=parent.raw
        self.dim=parent.dim
        self.variant,self.block_index=variant,block_index

    def forward(self,x,t,*,capture: list | None=None):
        context=self.raw.time(t)
        hidden=self.raw.input(x)
        for index,block in enumerate(self.raw.blocks):
            scale,shift=block.condition(context).chunk(2,dim=-1)
            normalized=block.norm(hidden)*(1+scale)+shift
            branch=block.ffn(normalized)
            selected=self.block_index is None or self.block_index==index
            if capture is not None or (selected and self.variant!="native"):
                centered,uniform,radial,tangent,beta=decompose(hidden,branch)
                if capture is not None:
                    capture.append(dict(block=index,hidden=hidden,branch=branch,centered=centered,
                        uniform=uniform,radial=radial,tangent=tangent,beta=beta,
                        step=block.residual_scale,epsilon=block.norm.eps))
                if selected:
                    if self.variant=="drop_uniform":branch=branch-uniform
                    elif self.variant=="drop_radial":branch=branch-radial
                    elif self.variant=="drop_tangent":branch=branch-tangent
            next_hidden=hidden+block.residual_scale*branch
            if capture is not None:
                capture[-1]["used_branch"]=branch
                capture[-1]["next_hidden"]=next_hidden
            hidden=next_hidden
        return self.raw.output(self.raw.norm(hidden))
