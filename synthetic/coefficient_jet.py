"""One independent second coefficient and its ordinary-map controls.

The quadratic arm is a Taylor jet of z'=(P+uC)z at internal time zero.
The implementation is not an exact solve or a full-map stability certificate.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from synthetic.models import OriginalODELayer


class CoefficientJetLayer(OriginalODELayer):
    def __init__(self, source: OriginalODELayer, mode: str = 'quadratic') -> None:
        nn.Module.__init__(self)
        if mode not in ('quadratic','affine','free','direct'):
            raise ValueError('unsupported second-coefficient control')
        required=dict(orders=2,gate_source='input',gate_transform='sigmoid',spectral_cap=None,
            clock_curvature=1.,rate_placement='after_projection',solve_return='map',
            operator_mode='multiplicative',forcing_mode='none')
        for key,value in required.items():
            if getattr(source,key)!=value: raise ValueError(f'original {key}={value} required')
        for key,value in source.__dict__.items():
            if not key.startswith('_'): setattr(self,key,value)
        self.t_func,self.proj=source.t_func,source.proj
        self.coefficient_mode=mode
        initial=(source.proj.weight.detach()@source.proj.weight.detach()
                 if mode=='direct' else torch.zeros_like(source.proj.weight))
        self.coefficient=nn.Parameter(initial)
        if mode=='free':
            self.free_weight=nn.Parameter(source.t_func[2].weight.detach().clone())
            self.free_bias=nn.Parameter(source.t_func[2].bias.detach().clone())

    def controls(self,x: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        hidden=self.t_func[1](self.t_func[0](x))
        s=self.duration_from_logit(self.t_func[2](hidden))
        r=self.duration_from_logit(F.linear(hidden,self.free_weight,self.free_bias)) if self.coefficient_mode=='free' else s
        return s,r

    def forward(self,x: torch.Tensor,context=None,*,return_duration: bool = False):
        if context is not None: raise ValueError('this study retains the original input clock')
        s,r=self.controls(x)
        first=F.linear(x,self.proj.weight)*s
        if self.coefficient_mode=='direct':
            output=x+first+.5*s.square()*F.linear(x,self.coefficient)
        else:
            # Preserve the original K2 operation order and all inherited gradients at C=0.
            second=F.linear(first,self.proj.weight)*s/2
            base=x+first+second
            clock=s if self.coefficient_mode=='quadratic' else (s.new_full(s.shape,.6) if self.coefficient_mode=='affine' else r)
            output=base+.5*s*clock*F.linear(x,self.coefficient)
        return (output,s) if return_duration else output

    def matrices(self) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        # B is a fixed reference for the affine/free controls, not their input-dependent effective coefficient.
        p=self.proj.weight
        if self.coefficient_mode=='direct':
            b=self.coefficient;c=b-p@p
        else:
            c=self.coefficient;b=p@p+c
        return p,b,c

    @torch.no_grad()
    def activation_stats(self,x: torch.Tensor,context=None) -> dict:
        s,r=self.controls(x);p,b,c=self.matrices()
        first=F.linear(x,p)*s
        second=F.linear(first,p)*s/2
        if self.coefficient_mode=='direct':
            extra=.5*s.square()*F.linear(x,b)-second
        else:
            clock=s if self.coefficient_mode=='quadratic' else (s.new_full(s.shape,.6) if self.coefficient_mode=='affine' else r)
            extra=.5*s*clock*F.linear(x,c)
        rms=lambda v:float(v.square().mean().sqrt())
        denominator=max(rms(x),1e-12)
        return dict(s_mean=float(s.mean()),s_std=float(s.std(unbiased=False)),s_min=float(s.min()),s_max=float(s.max()),
            r_mean=float(r.mean()),r_std=float(r.std(unbiased=False)),r_minus_s_rms=rms(r-s),
            gate_channels=1,input_rms=rms(x),coefficient_mode=self.coefficient_mode,
            term1_rms_over_input=rms(first)/denominator,term2_rms_over_input=rms(second)/denominator,
            added_action_rms_over_input=rms(extra)/denominator,
            output_rms_over_input=rms(self(x))/denominator,
            C_frobenius=float(c.norm()),reference_B_frobenius=float(b.norm()),
            P_C_commutator_frobenius=float((p@c-c@p).norm()))
