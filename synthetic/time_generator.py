"""Original K2 solves with one low-rank, externally time-dependent generator."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from synthetic.models import OriginalODELayer
from synthetic.recurrent_depth import RecurrentDepthNet, build as build_parent


def time_coordinate(t: torch.Tensor) -> torch.Tensor:
    """Zero mean and unit second moment under the fixed uniform-time training law."""
    return math.sqrt(12)*(t-.5)


class LowRankTime(nn.Module):
    def __init__(self, input_width: int, output_width: int, mode: str):
        super().__init__()
        if mode not in ("static","time"):
            raise ValueError("unsupported adapter mode")
        self.mode=mode
        self.down=nn.Linear(input_width,4,bias=False)
        self.up=nn.Linear(4,output_width,bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        value=self.up(self.down(x))
        return value*time_coordinate(t) if self.mode=="time" else value

    def matrix(self):
        return self.up.weight@self.down.weight


class TimeODELayer(OriginalODELayer):
    def __init__(self, original: OriginalODELayer, mode: str):
        if original.orders!=2 or original.gate_source!="input" or original.spectral_cap is not None:
            raise ValueError("the prototype preserves the original scalar K2 solve")
        width=original.proj.in_features
        super().__init__(width,original.t_func[0].out_features,orders=2,
                         tau=original.tau,scale=original.scale,shift=original.shift)
        self.load_state_dict(original.state_dict())
        self.adapter=LowRankTime(width,width,mode)

    def project(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return F.linear(x,self.proj.weight)+self.adapter(x,t)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, *, return_duration=False):
        if context is None or context.shape!=(len(x),1):
            raise ValueError("external time must have shape[batch,1]")
        s=self.step_size(x)
        total,term=x,x
        for order in (1,2):
            term=self.project(term,context)*s/order
            total=total+term
        return (total,s) if return_duration else total

    @torch.no_grad()
    def activation_stats(self,x,context=None):
        s=self.step_size(x)
        first=self.project(x,context)*s
        second=self.project(first,context)*s/2
        rms=lambda z:float(z.square().mean().sqrt())
        denominator=max(rms(x),1e-12)
        return dict(s_mean=float(s.mean()),s_std=float(s.std(unbiased=False)),
            s_min=float(s.min()),s_max=float(s.max()),input_rms=rms(x),
            term1_rms_over_input=rms(first)/denominator,term2_rms_over_input=rms(second)/denominator,
            output_rms_over_input=rms(x+first+second)/denominator,
            adapter_delta_frobenius=float(self.adapter.matrix().norm()),
            external_time_coordinate_mean=float(time_coordinate(context).mean()),
            operator_mode="P0+phi(t)*UV" if self.adapter.mode=="time" else "P0+UV")


class TimeSwiGLU(nn.Module):
    def __init__(self,original,mode: str):
        super().__init__()
        self.w12,self.w3=original.w12,original.w3
        self.adapter12=LowRankTime(self.w12.in_features,self.w12.out_features,mode)
        self.adapter3=LowRankTime(self.w3.in_features,self.w3.out_features,mode)

    def forward(self,x,t):
        gate,value=(self.w12(x)+self.adapter12(x,t)).chunk(2,dim=-1)
        hidden=F.silu(gate)*value
        return self.w3(hidden)+self.adapter3(hidden,t)


class TimeGeneratorNet(RecurrentDepthNet):
    def __init__(self,parent: RecurrentDepthNet,family: str,variant: str):
        if variant not in ("none","static","time"):
            raise ValueError("unsupported generator variant")
        super().__init__(parent.raw,"native")
        self.family,self.variant=family,variant
        if variant!="none":
            for block in self.raw.blocks:
                if family=="ode_k2":
                    block.ffn.ode1=TimeODELayer(block.ffn.ode1,variant)
                    block.ffn.ode2=TimeODELayer(block.ffn.ode2,variant)
                elif family=="swiglu192":
                    block.ffn=TimeSwiGLU(block.ffn,variant)
                else:
                    raise ValueError("only the compute-matched generic family is adapted")

    def forward(self,x,t,*,return_all=False,updates=3,step_size=None):
        if self.variant=="none":
            return super().forward(x,t,return_all=return_all,updates=updates,step_size=step_size)
        if x.ndim!=2 or x.shape[1]!=self.dim or t.shape!=(len(x),1):
            raise ValueError("expected state[batch,dim] and external time[batch,1]")
        if updates!=3 or step_size is not None:
            raise ValueError("this prototype fixes the original three independent updates")
        context=self.raw.time(t)
        hidden=self.raw.input(x)
        predictions=[]
        for index,block in enumerate(self.raw.blocks):
            scale,shift=block.condition(context).chunk(2,dim=-1)
            z=block.norm(hidden)*(1+scale)+shift
            hidden=hidden+block.residual_scale*block.ffn(z,t)
            if return_all or index==2:
                predictions.append(self.raw.output(self.raw.norm(hidden)))
        return predictions if return_all else predictions[-1]


def build(dim: int,family: str,mode: str):
    if family not in ("ode_k2","swiglu192"):
        raise ValueError("unsupported family")
    return TimeGeneratorNet(build_parent(dim,family,"native"),family,mode)
