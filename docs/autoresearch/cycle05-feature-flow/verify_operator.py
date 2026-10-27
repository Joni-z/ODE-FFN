"""Independent algebra, complete derivative and numerical-integration fixtures."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def slope(x: torch.Tensor) -> torch.Tensor:
    s=torch.sigmoid(x)
    return s*(1+x*(1-s))


def mean_gate(a: torch.Tensor,b: torch.Tensor,count: int) -> torch.Tensor:
    nodes,weights=np.polynomial.legendre.leggauss(count)
    return sum(float(w/2)*F.silu(a+float((s+1)/2)*b) for s,w in zip(nodes,weights))


def verify(quadrature,midpoint) -> dict:
    torch.manual_seed(0);torch.set_num_threads(1)
    w12,w3=nn.Linear(3,8).double(),nn.Linear(4,3).double()
    model=quadrature(w12,w3)
    with torch.no_grad():model.transport.copy_(.3*torch.cos(torch.arange(4,dtype=torch.float64)).reshape(2,2))
    x=torch.sin(torch.arange(9,dtype=torch.float64)).reshape(3,3).requires_grad_(True)
    a,v=model.w12(x).chunk(2,-1);p=a.new_zeros(4,4);p[:2,2:]=model.transport
    assert torch.equal(p@p,torch.zeros_like(p))
    eye=torch.eye(4,dtype=torch.float64);nodes=model.nodes
    g=sum(.5*F.silu(F.linear(a,eye+s*p)) for s in nodes)
    literal=model.w3(v*g);direct=model(x)
    torch.testing.assert_close(literal,direct,rtol=1e-12,atol=1e-12)
    coefficients=torch.cos(torch.arange(9,dtype=torch.float64)).reshape(3,3)
    grads=[torch.autograd.grad((y*coefficients).sum()+.03*y.square().sum(),(x,*model.parameters()),retain_graph=True) for y in (literal,direct)]
    maximum=0.
    for aa,bb in zip(*grads):
        torch.testing.assert_close(aa,bb,rtol=1e-11,atol=1e-12)
        maximum=max(maximum,float((aa-bb).abs().max()))
    wg,vv=model.w12.weight.chunk(2,0)
    jg=sum(.5*torch.diag_embed(slope(F.linear(a,eye+s*p)))@(eye+s*p)@wg for s in nodes)
    analytic=model.w3.weight@(torch.diag_embed(v)@jg+torch.diag_embed(g)@vv)
    actual=torch.stack([torch.autograd.grad(direct[:,i].sum(),x,retain_graph=True)[0] for i in range(3)],1)
    torch.testing.assert_close(analytic,actual,rtol=1e-11,atol=1e-12)
    receiving=(coefficients+.06*direct)@model.w3.weight
    delta=F.linear(a[:,2:],model.transport)
    gc=sum(.5*s*((receiving[:,:2]*v[:,:2])*slope(a[:,:2]+s*delta)).T@a[:,2:] for s in nodes)
    gc_actual=torch.autograd.grad((direct*coefficients).sum()+.03*direct.square().sum(),model.transport,retain_graph=True)[0]
    torch.testing.assert_close(gc,gc_actual,rtol=1e-11,atol=1e-12)
    mid=midpoint(w12,w3)
    with torch.no_grad():mid.transport.copy_(model.transport)
    absorbed=F.linear(a,eye+p/2)
    torch.testing.assert_close(mid(x),mid.w3(F.silu(absorbed)*v),rtol=1e-12,atol=1e-12)
    folded_w=(eye+p/2)@wg;folded_b=(eye+p/2)@model.w12.bias[:4]
    torch.testing.assert_close(F.linear(x,folded_w,folded_b),absorbed,rtol=1e-12,atol=1e-12)
    # Values and ALL local derivatives, including C, agree at zero transport.
    q0,m0=quadrature(w12,w3),midpoint(w12,w3)
    torch.testing.assert_close(q0(x),m0(x),rtol=0,atol=0)
    native=w3(F.silu(w12(x).chunk(2,-1)[0])*w12(x).chunk(2,-1)[1])
    torch.testing.assert_close(q0(x),native,rtol=0,atol=0)
    common=[]
    for module in (q0,m0):common.append(torch.autograd.grad(module(x).square().sum(),(x,*module.parameters()),retain_graph=True))
    for aa,bb in zip(*common):torch.testing.assert_close(aa,bb,rtol=1e-12,atol=1e-12)
    # Full joint RK4 state independently evolves a, not the closed-form path.
    initial=a.detach();value=v.detach();state=torch.cat((initial,torch.zeros_like(initial)),-1)
    def field(z):
        aa=z[:,:4]
        return torch.cat((F.linear(aa,p.detach()),value*F.silu(aa)),-1)
    step=1/128
    for _ in range(128):
        k1=field(state);k2=field(state+step*k1/2);k3=field(state+step*k2/2);k4=field(state+step*k3)
        state=state+step*(k1+2*k2+2*k3+k4)/6
    reference=mean_gate(initial,F.linear(initial,p.detach()),64)
    torch.testing.assert_close(state[:,:4],initial+F.linear(initial,p.detach()),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(state[:,4:],value*reference,rtol=1e-10,atol=1e-11)
    # Nilpotent pointwise Jacobian does not terminate nonlinear time derivatives.
    jac=a.new_zeros(8,8);jac[:4,:4]=p.detach();jac[4:,:4]=torch.diag(value[0]*slope(initial[0]))
    assert torch.equal(jac@jac@jac,torch.zeros_like(jac))
    s=initial[0].requires_grad_(True);sigma2=torch.autograd.grad(slope(s).sum(),s)[0]
    third=value[0]*sigma2*F.linear(initial[0],p.detach()).square()
    assert float(third.abs().max())>1e-8
    # A single receiver contains two independent affine ridge directions.
    def witness(z):return sum(.5*F.silu(z[0]+t*z[1]) for t in nodes)
    hess=torch.autograd.functional.hessian(witness,torch.zeros(2,dtype=torch.float64))
    expected_hess=torch.tensor([[.5,.25],[.25,1/6]],dtype=torch.float64)
    torch.testing.assert_close(hess,expected_hess,rtol=1e-12,atol=1e-12)
    assert torch.linalg.matrix_rank(hess)==2
    # Independent high-order references and a conservative global remainder.
    agrid=torch.linspace(-12,12,101,dtype=torch.float64)[:,None]
    bgrid=torch.linspace(-6,6,31,dtype=torch.float64)[None,:]
    q2=sum(.5*F.silu(agrid+t*bgrid) for t in nodes)
    q32=mean_gate(agrid,bgrid,32);q64=mean_gate(agrid,bgrid,64)
    torch.testing.assert_close(q32,q64,rtol=1e-12,atol=1e-12)
    bound=bgrid.pow(4)*(1+2/math.e)/4320
    assert bool(((q2-q64).abs()<=bound+2e-14).all())
    # Formula for the fourth activation derivative, checked by nested AD.
    z=torch.linspace(-20,20,401,dtype=torch.float64).requires_grad_(True);d=F.silu(z)
    for _ in range(4):d=torch.autograd.grad(d.sum(),z,create_graph=True)[0]
    prob=torch.sigmoid(z);fourth=prob*(1-prob)*(4*(1-6*prob+6*prob.square())+z*(1-2*prob)*(1-12*prob+12*prob.square()))
    torch.testing.assert_close(d,fourth,rtol=1e-9,atol=1e-12)
    assert float(fourth.detach().abs().max())<=1+2/math.e
    return dict(status='PASS',literal_value_max=float((literal-direct).detach().abs().max()),
        full_gradient_max=maximum,complete_jacobian_max=float((analytic-actual).detach().abs().max()),
        C_gradient_max=float((gc-gc_actual).detach().abs().max()),midpoint_absorbs=True,
        zero_values_and_all_tangents_pair=True,receiver_hessian_determinant=float(torch.det(hess)),
        pointwise_J_cube_zero=True,nonlinear_third_time_derivative_max=float(third.detach().abs().max()),
        rk4_integral_max=float((state[:,4:]-value*reference).abs().max()),
        quadrature32_vs64_max=float((q32-q64).abs().max()),global_remainder_bound_passed=True,
        receiver_witness_scope='single feature, not whole-network expressivity')
