"""Literal exponential, complete derivatives and fixed-dictionary witness."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def verify(transport,diagonal) -> dict:
    torch.manual_seed(0);torch.set_num_threads(1)
    w12,w3=nn.Linear(3,8).double(),nn.Linear(4,3).double()
    model=transport(w12,w3)
    with torch.no_grad():model.transport.copy_(torch.cos(torch.arange(4,dtype=torch.float64)).reshape(2,2)*.3)
    x=torch.sin(torch.arange(9,dtype=torch.float64)).reshape(3,3).requires_grad_(True)
    activation,value=model.w12(x).chunk(2,-1);gate=F.silu(activation)
    z=model.transport.new_zeros(2,2)
    pc=torch.cat([torch.cat([z,model.transport],-1),torch.cat([z,z],-1)],-2)
    zb=torch.zeros(3,4,4,dtype=torch.float64)
    kk=torch.cat([torch.cat([pc.expand(3,4,4),zb],-1),torch.cat([torch.diag_embed(value),zb],-1)],-2)
    assert torch.equal(kk@kk@kk,torch.zeros_like(kk))
    identity=torch.eye(8,dtype=torch.float64).expand(3,8,8)
    endpoint=torch.matrix_exp(kk)
    torch.testing.assert_close(endpoint,identity+kk+.5*(kk@kk),rtol=1e-12,atol=1e-12)
    initial=torch.cat([gate,torch.zeros_like(gate)],-1)
    literal=model.w3(torch.bmm(endpoint,initial.unsqueeze(-1)).squeeze(-1)[:,4:])
    direct=model(x);torch.testing.assert_close(literal,direct,rtol=1e-12,atol=1e-12)
    coefficient=torch.cos(torch.arange(9,dtype=torch.float64)).reshape(3,3)
    derivatives=[]
    for out in (literal,direct):
        derivatives.append(torch.autograd.grad((out*coefficient).sum()+.03*out.square().sum(),(x,*model.parameters()),retain_graph=True))
    gradient_max=0.
    for a,b in zip(*derivatives):
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11);gradient_max=max(gradient_max,float((a-b).abs().max()))
    wg,vv=model.w12.weight.chunk(2,0);mix=torch.eye(4,dtype=torch.float64)+pc/2
    probability=torch.sigmoid(activation);slope=probability*(1+activation*(1-probability))
    first=torch.diag_embed(value)@mix@torch.diag_embed(slope)@wg
    second=torch.diag_embed(F.linear(gate,mix))@vv
    expected=model.w3.weight@(first+second)
    actual=torch.stack([torch.autograd.grad(direct[:,i].sum(),x,retain_graph=True)[0] for i in range(3)],1)
    torch.testing.assert_close(expected,actual,rtol=1e-10,atol=1e-11)
    delta=coefficient+.06*direct
    expected_c=.5*((delta@model.w3.weight[:,:2])*value[:,:2]).T@gate[:,2:]
    actual_c=torch.autograd.grad((direct*coefficient).sum()+.03*direct.square().sum(),model.transport,retain_graph=True)[0]
    torch.testing.assert_close(expected_c,actual_c,rtol=1e-10,atol=1e-11)
    # Zero transport/scaling preserves all common local parameter responses.
    for candidate in (transport(w12,w3),diagonal(w12,w3)):
        ordinary=lambda state:w3(F.silu(w12(state).chunk(2,-1)[0])*w12(state).chunk(2,-1)[1])
        assert torch.equal(candidate(x),ordinary(x))
        common=[x,candidate.w12.weight,candidate.w12.bias,candidate.w3.weight,candidate.w3.bias]
        old=[x,w12.weight,w12.bias,w3.weight,w3.bias]
        a=torch.autograd.grad(candidate(x).square().sum(),common,retain_graph=True)
        b=torch.autograd.grad(ordinary(x).square().sum(),old,retain_graph=True)
        for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,rtol=1e-12,atol=1e-12)
    # Fixed dictionary g1=SiLU(h1), g2=SiLU(h2), with full-rank A_g2.
    witness=transport(nn.Linear(2,4,bias=False),nn.Linear(2,2,bias=False)).double()
    with torch.no_grad():
        witness.w12.weight.copy_(torch.cat([torch.eye(2),torch.eye(2)],0));witness.w3.weight.copy_(torch.eye(2));witness.transport.fill_(2.)
    points=torch.tensor([[.3,.8],[-.7,.2],[1.,-.4]],dtype=torch.float64)
    gg=F.silu(points);truth=torch.stack([points[:,0]*(gg[:,0]+gg[:,1]),points[:,1]*gg[:,1]],-1)
    torch.testing.assert_close(witness(points),truth,rtol=1e-12,atol=1e-12)
    rank=int(torch.linalg.matrix_rank(torch.eye(2)))
    return dict(status='PASS',literal_value_max=float((literal-direct).abs().max().detach()),
        parameter_and_input_gradient_max=gradient_max,complete_jacobian_max=float((expected-actual).abs().max().detach()),
        C_gradient_max=float((expected_c-actual_c).abs().max().detach()),joint_cube_zero=True,
        zero_transport_and_scaling_equal_native=True,fixed_dictionary_coefficient_rank=rank,
        rank_scope='fixed two-feature dictionary, not a whole-network lower bound')
