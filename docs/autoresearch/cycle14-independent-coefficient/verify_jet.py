"""Independent complete derivatives, initialization and internal Taylor fixtures."""
from __future__ import annotations

import copy

import numpy as np
import torch


def verify(operator,base_layer) -> dict:
    torch.manual_seed(1702)
    source=base_layer(5,7).double();x=torch.linspace(-.7,.8,15,dtype=torch.float64).reshape(3,5)
    maximum=0.;initial=[]
    for mode in ('quadratic','affine','free'):
        reference=copy.deepcopy(source);rng=torch.get_rng_state().clone()
        candidate=operator.CoefficientJetLayer(copy.deepcopy(source),mode)
        assert torch.equal(rng,torch.get_rng_state())
        a=x.clone().requires_grad_(True);b=x.clone().requires_grad_(True)
        left=reference(a);right=candidate(b);torch.testing.assert_close(left,right,atol=0,rtol=0)
        names=list(reference.named_parameters());new=dict(candidate.named_parameters())
        l=torch.autograd.grad(left.square().sum(),(a,*(p for _,p in names)))
        r=torch.autograd.grad(right.square().sum(),(b,*(new[n] for n,_ in names)))
        for u,v in zip(l,r): torch.testing.assert_close(u,v,atol=0,rtol=0)
        initial.append(dict(mode=mode,values_and_inherited_gradients_bitwise=True,rng_unchanged=True))
    direct=operator.CoefficientJetLayer(copy.deepcopy(source),'direct')
    torch.testing.assert_close(direct(x),source(x),atol=1e-12,rtol=1e-12)
    for mode in ('quadratic','affine','free','direct'):
        layer=operator.CoefficientJetLayer(copy.deepcopy(source),mode)
        with torch.no_grad(): layer.coefficient.copy_(torch.linspace(-.1,.12,25,dtype=torch.float64).reshape(5,5))
        z=x[0].clone().requires_grad_(True);s,r=layer.controls(z[None]);s=s.squeeze();r=r.squeeze()
        ds=torch.autograd.grad(s,z,retain_graph=True)[0];dr=torch.autograd.grad(r,z,retain_graph=True)[0]
        p=layer.proj.weight;coefficient=layer.coefficient
        if mode=='direct':
            matrix=torch.eye(5,dtype=z.dtype)+s*p+.5*s*s*coefficient
            expected=matrix+torch.outer((p+s*coefficient)@z,ds)
        else:
            matrix=torch.eye(5,dtype=z.dtype)+s*p+.5*s*s*(p@p)
            vector=(p+s*p@p)@z
            if mode=='quadratic':
                matrix=matrix+.5*s*s*coefficient;vector=vector+s*coefficient@z
                expected=matrix+torch.outer(vector,ds)
            elif mode=='affine':
                matrix=matrix+.3*s*coefficient;vector=vector+.3*coefficient@z
                expected=matrix+torch.outer(vector,ds)
            else:
                matrix=matrix+.5*s*r*coefficient
                expected=matrix+torch.outer(vector+.5*r*coefficient@z,ds)+torch.outer(.5*s*coefficient@z,dr)
        actual=torch.autograd.functional.jacobian(lambda value:layer(value[None])[0],z)
        error=float((actual-expected).detach().abs().max());maximum=max(maximum,error);assert error<1e-10
        q=torch.linspace(-.4,.6,5,dtype=z.dtype)
        gp,gc=torch.autograd.grad((layer(z[None])[0]*q).sum(),(p,coefficient))
        expected_p=s*torch.outer(q,z)
        if mode!='direct': expected_p=expected_p+.5*s*s*(torch.outer(q,p@z)+torch.outer(p.T@q,z))
        factor={'quadratic':.5*s*s,'affine':.3*s,'free':.5*s*r,'direct':.5*s*s}[mode]
        torch.testing.assert_close(gp,expected_p,atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(gc,factor*torch.outer(q,z),atol=1e-10,rtol=1e-10)
    p=np.array([[-.3,0.],[0.,.5]]);c=np.array([[0.,1.],[-.2,0.]])
    a,b=.3,.8
    curve=lambda s:np.eye(2)+s*p+s*s*(p@p+c)/2
    difference=curve(a)@curve(b)-curve(b)@curve(a)
    predicted=.5*a*b*(b-a)*(p@c-c@p)
    np.testing.assert_allclose(difference,predicted,atol=1e-14,rtol=1e-12)
    assert np.linalg.norm(difference)>1e-3
    # Independent RK4 for z'=(P+uC)z, with a fixed high resolution per short interval.
    vector=np.array([.7,-.4]);errors=[]
    for duration in (.04,.02,.01):
        state=vector.copy();h=duration/1024
        for i in range(1024):
            u=i*h;field=lambda t,y:(p+t*c)@y
            k1=field(u,state);k2=field(u+h/2,state+h*k1/2)
            k3=field(u+h/2,state+h*k2/2);k4=field(u+h,state+h*k3)
            state+=h*(k1+2*k2+2*k3+k4)/6
        jet=curve(duration)@vector;errors.append(float(np.linalg.norm(state-jet)))
        leading=duration**3*(p@p@p+p@c+2*c@p)@vector/6
        np.testing.assert_allclose(state-jet,leading,atol=duration**4,rtol=.1)
    ratios=[errors[i]/errors[i+1] for i in range(2)];assert all(7.5<r<8.5 for r in ratios)
    return dict(status='PASS',initial=initial,direct_float64_equivalence=True,
        active_complete_Jacobian_max=maximum,active_P_coefficient_gradients=True,
        commutator_witness=True,small_duration_error_ratios=ratios)
