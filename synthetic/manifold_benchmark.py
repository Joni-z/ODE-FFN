"""Paired embedded-mixture laws and fixed prediction heads for a new benchmark.

The basis belongs to data generation/evaluation. It is never a model input.
Existing synthetic laws, objectives and evaluation code are not modified.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

DIM=128
LATENT_DIM=2
VARIANCE=.04
CAP=.05


def make_basis() -> np.ndarray:
    rng=np.random.default_rng(np.random.SeedSequence([0,20261028,701]))
    q,r=np.linalg.qr(rng.standard_normal((DIM,LATENT_DIM)))
    return q*np.sign(np.diag(r))[None,:]


def paired_targets(rng: np.random.Generator,count: int,basis: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    component=rng.integers(0,8,size=count);angles=component*2*np.pi/8
    latent=3*np.stack((np.cos(angles),np.sin(angles)),-1)+.2*rng.standard_normal((count,2))
    thin=latent@basis.T
    noise=rng.standard_normal((count,DIM));normal=noise-(noise@basis)@basis.T
    return thin.astype(np.float32),(thin+.2*normal).astype(np.float32)


def oracle(x: np.ndarray,t: np.ndarray,basis: np.ndarray,normal_variance: float,*,jacobian: bool=False) -> dict:
    x=np.asarray(x,dtype=np.float64);t=np.asarray(t,dtype=np.float64).reshape(-1,1)
    if x.shape!=(len(t),DIM) or np.any((t<0)|(t>1)):raise ValueError('invalid x/t shapes or times')
    a=1-t;delta=np.maximum(a,CAP);z=x@basis
    angles=np.arange(8)*2*np.pi/8;means=3*np.stack((np.cos(angles),np.sin(angles)),-1)
    d=a*a+t*t*VARIANCE;residual=z[:,None,:]-t[:,:,None]*means[None]
    logits=-np.sum(residual**2,axis=-1)/(2*d);logits-=logits.max(1,keepdims=True)
    p=np.exp(logits);p/=p.sum(1,keepdims=True);mean=p@means
    coefficient=t*VARIANCE/d;mix=a*a/d
    latent_mean=coefficient*z+mix*mean
    dn=a*a+t*t*normal_variance
    cn=np.divide(t*normal_variance,dn,out=np.zeros_like(dn),where=dn>0)
    normal=x-z@basis.T;clean=latent_mean@basis.T+cn*normal
    decoded=(clean-x)/delta
    result=dict(oracle=decoded,clean_mean=clean,normal_gain=cn,
        direct_normal_coefficient=(cn-1)/delta,skip_residual_coefficient=cn/delta,normal_path_variance=dn)
    if jacobian:
        centered=means[None]-mean[:,None]
        covariance=np.einsum('bk,bki,bkj->bij',p,centered,centered)
        jlatent=coefficient[:,:,None]*np.eye(2)[None]+(a*a*t/(d*d))[:,:,None]*covariance
        projector=basis@basis.T
        jm=np.einsum('di,bij,ej->bde',basis,jlatent,basis)+cn[:,:,None]*(np.eye(DIM)-projector)[None]
        result.update(clean_jacobian=jm,oracle_jacobian=(jm-np.eye(DIM)[None])/delta[:,:,None])
    return result


def bottleneck_bounds(t: np.ndarray,normal_variance: float,width: int=64) -> dict:
    t=np.asarray(t,dtype=np.float64).reshape(-1,1);a=1-t;delta=np.maximum(a,CAP)
    d=a*a+t*t*normal_variance
    gain=np.divide(t*normal_variance,d,out=np.zeros_like(d),where=d>0)
    beta=(gain-1)/delta;gamma=gain/delta
    unobserved=max(DIM-width-LATENT_DIM,0)
    jacobian_missing=max(DIM-LATENT_DIM-(width-1),0)
    return dict(direct_field_population_floor=unobserved/DIM*beta**2*d,
        skip_field_population_floor=unobserved/DIM*gamma**2*d,
        direct_J_floor=jacobian_missing*beta**2,skip_J_floor=jacobian_missing*gamma**2)


class CappedHead(nn.Module):
    """One raw native network, three explicit decoders, no new parameters."""

    def __init__(self,raw: nn.Module,head: str) -> None:
        super().__init__()
        if head not in ('clean','shifted','direct'):raise ValueError('unknown prediction head')
        self.raw,self.head,self.dim,self.mode=raw,head,DIM,'native'

    def forward(self,x: torch.Tensor,t: torch.Tensor,*,updates: int=3,step_size=None) -> torch.Tensor:
        if updates!=3 or step_size is not None:raise ValueError('this benchmark keeps three native blocks')
        out=self.raw(x,t);delta=(1-t).clamp_min(CAP)
        if self.head=='clean':return (out-x)/delta
        if self.head=='shifted':return out-x/delta
        return out

    def ode_layers(self):yield from self.raw.ode_layers()


def regression_target(target: torch.Tensor,source: torch.Tensor,t: torch.Tensor) -> torch.Tensor:
    return (1-t)/(1-t).clamp_min(CAP)*(target-source)
