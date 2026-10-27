"""Exact full-component-support mixture of independent and Hadamard signs."""
from __future__ import annotations

import numpy as np


class FullSupportMixture:
    def __init__(self,locations,rotation,signs):
        self.locations=np.asarray(locations,dtype=np.float64)
        self.rotation=np.asarray(rotation,dtype=np.float64)
        self.signs=np.asarray(signs,dtype=np.float64)
        self.dim=len(self.locations)
        self.means=self.signs*self.locations
        self.variance=.04
        self.prior_weight=.5

    def sample(self,rng,count):
        branch=rng.random(count)<self.prior_weight
        independent=rng.integers(0,2,size=(count,self.dim))*2-1
        chosen=self.signs[rng.integers(len(self.signs),size=count)]
        signs=np.where(branch[:,None],chosen,independent)
        return ((signs*self.locations+.2*rng.standard_normal((count,self.dim)))@self.rotation.T).astype(np.float32)

    def oracle(self,x,time,jacobian=False):
        q=np.asarray(x,dtype=np.float64)@self.rotation
        t=np.asarray(time,dtype=np.float64)
        if t.ndim==0:t=np.full((len(q),1),t)
        if t.ndim==1:t=t[:,None]
        assert q.shape==(len(q),self.dim) and t.shape==(len(q),1)
        d=(1-t)**2+self.variance*t**2
        alpha=(self.variance*t-(1-t))/d;beta=(1-t)/d;kappa=t/d
        coordinate=kappa*q*self.locations
        m0=self.locations*np.tanh(coordinate)
        logz0=(np.logaddexp(coordinate,-coordinate)-np.log(2)).sum(-1,keepdims=True)
        logits=(kappa*q)@self.means.T
        maximum=logits.max(-1,keepdims=True)
        exponent=np.exp(logits-maximum)
        probability=exponent/exponent.sum(-1,keepdims=True)
        logz1=maximum+np.log(exponent.mean(-1,keepdims=True))
        logodds=np.log(self.prior_weight/(1-self.prior_weight))+logz1-logz0
        weight=np.exp(-np.logaddexp(0,-logodds))
        m1=probability@self.means
        mean=(1-weight)*m0+weight*m1
        velocity=(alpha*q+beta*mean)@self.rotation.T
        if not jacobian:return velocity
        diagonal=self.locations**2*(1-np.tanh(coordinate)**2)
        c0=diagonal[:,:,None]*np.eye(self.dim)
        c1=np.einsum('bk,ki,kj->bij',probability,self.means,self.means,optimize=True)-m1[:,:,None]*m1[:,None,:]
        delta=m1-m0
        cov=(1-weight[:,:,None])*c0+weight[:,:,None]*c1+weight[:,:,None]*(1-weight[:,:,None])*delta[:,:,None]*delta[:,None,:]
        latent=alpha[:,:,None]*np.eye(self.dim)+(t*(1-t)/d**2)[:,:,None]*cov
        physical=np.einsum('ij,bjk,lk->bil',self.rotation,latent,self.rotation,optimize=True)
        return velocity,physical,weight

    def specification(self):
        counts=np.array([32,2**self.dim-32],dtype=np.float64)
        weights=np.array([.5/32+.5/2**self.dim,.5/2**self.dim])
        return dict(kind='half_product_half_sign_code',dim=self.dim,locations=self.locations.tolist(),
            rotation=self.rotation.tolist(),signs=self.signs.tolist(),variance=self.variance,
            sign_code_prior_weight=self.prior_weight,positive_component_count=2**self.dim,
            distinguished_component_count=32,component_label_entropy_bits=float(-np.sum(counts*weights*np.log2(weights))))
