"""Dense Gaussian conditioning, AD and endpoint checks independent of factorization."""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn


def verify(core) -> dict:
    torch.set_num_threads(1);q=core.make_basis();np.testing.assert_allclose(q.T@q,np.eye(2),rtol=1e-12,atol=1e-12)
    basis=torch.from_numpy(q);projector=basis@basis.T;identity=torch.eye(core.DIM,dtype=torch.float64)
    angles=torch.arange(8,dtype=torch.float64)*2*torch.pi/8
    means=3*torch.stack((torch.cos(angles),torch.sin(angles)),-1)@basis.T
    xx=torch.sin(torch.arange(7*core.DIM,dtype=torch.float64).reshape(7,core.DIM))*.4
    tt=torch.tensor([.03,.1,.3,.5,.7,.9,.97],dtype=torch.float64)[:,None]
    max_value=max_jac=0.
    for normal_variance in (0.,.04):
        covariance=.04*projector+normal_variance*(identity-projector)
        a=1-tt;ct=a[:,:,None]**2*identity+tt[:,:,None]**2*covariance
        precision=torch.linalg.inv(ct)
        def dense(x):
            residual=x[:,None,:]-tt[:,:,None]*means[None]
            logits=-.5*torch.einsum('bki,bij,bkj->bk',residual,precision,residual)
            weights=torch.softmax(logits,1)
            component=means[None]+tt[:,:,None]*torch.einsum('ij,bjl,bkl->bki',covariance,precision,residual)
            posterior=torch.sum(weights[:,:,None]*component,1)
            return (posterior-x)/(1-tt).clamp_min(.05)
        x=xx.detach().clone().requires_grad_(True);prediction=dense(x)
        jac=torch.stack([torch.autograd.grad(prediction[:,j].sum(),x,retain_graph=True)[0] for j in range(core.DIM)],1)
        expected=core.oracle(xx.numpy(),tt.numpy(),q,normal_variance,jacobian=True)
        np.testing.assert_allclose(prediction.detach().numpy(),expected['oracle'],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(jac.detach().numpy(),expected['oracle_jacobian'],rtol=1e-9,atol=1e-10)
        max_value=max(max_value,float(np.max(np.abs(prediction.detach().numpy()-expected['oracle']))))
        max_jac=max(max_jac,float(np.max(np.abs(jac.detach().numpy()-expected['oracle_jacobian']))))
        endpoint=core.oracle(xx[:2].numpy(),np.array([[0.],[1.]]),q,normal_variance,jacobian=True)
        np.testing.assert_allclose(endpoint['clean_mean'][0],np.zeros(core.DIM),atol=1e-12)
        truth=xx[1].numpy() if normal_variance else (xx[1].numpy()@q)@q.T
        np.testing.assert_allclose(endpoint['clean_mean'][1],truth,rtol=1e-12,atol=1e-12)
        if normal_variance==0:
            jm=expected['clean_jacobian']
            assert max(np.linalg.matrix_rank(matrix,tol=1e-9) for matrix in jm)==2
            np.testing.assert_allclose(jm@(np.eye(core.DIM)-q@q.T),0,atol=1e-11)
    # The two skip heads pair values but have the distinct declared time scaling.
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.linear=nn.Linear(core.DIM,core.DIM,bias=False).double()
            with torch.no_grad():self.linear.weight.zero_()
        def forward(self,x,t):return self.linear(x)
    raw=Tiny();heads={name:core.CappedHead(copy.deepcopy(raw),name) for name in ('clean','shifted','direct')}
    torch.testing.assert_close(heads['clean'](xx,tt),heads['shifted'](xx,tt),rtol=0,atol=0)
    assert torch.equal(heads['direct'](xx,tt),torch.zeros_like(xx))
    weight=torch.sin(torch.arange(core.DIM**2,dtype=torch.float64).reshape(core.DIM,core.DIM))*.01
    dx=xx@weight.T;delta=(1-tt).clamp_min(.05)
    for name,model in heads.items():
        _,tangent=torch.autograd.functional.jvp(lambda w:torch.func.functional_call(model,{'raw.linear.weight':w},(xx,tt)),raw.linear.weight,weight)
        torch.testing.assert_close(tangent,dx/delta if name=='clean' else dx,rtol=1e-11,atol=1e-11)
    # Deterministic normal basis realizes the predicted missing-coordinate risk.
    normal=np.eye(core.DIM)-q@q.T;eig,axes=np.linalg.eigh(normal);normal_axes=axes[:,eig>.5]
    observed=normal_axes[:,:62];x=np.sqrt(126*.25)*normal_axes.T;t=np.full((126,1),.5)
    target=core.oracle(x,t,q,0.)['oracle'];prediction=-2*(x@observed)@observed.T
    risk=np.mean((prediction-target)**2);np.testing.assert_allclose(risk,64/core.DIM,rtol=1e-10,atol=1e-10)
    assert risk>=float(core.bottleneck_bounds(np.array([[.5]]),0.)['direct_field_population_floor'][0,0])
    thin_floor=62/128*(29/30)
    return dict(status='PASS',dense_conditioning_value_max=max_value,dense_AD_jacobian_max=max_jac,
        thin_posterior_response_rank=2,endpoint_extensions_pass=True,skip_initial_values_pair=True,
        head_parameter_tangents_checked=True,normal_basis_risk=float(risk),thin_direct_uniform_population_floor=thin_floor,
        bounds_are_population_not_finite_bank=True)
