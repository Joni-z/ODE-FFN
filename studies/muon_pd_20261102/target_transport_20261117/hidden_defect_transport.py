"""Propagate both prescribed hidden defects through the unchanged model forward."""
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch.autograd import forward_ad
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'coefficient_20261116'))
from rk4_step import rk4_step_func  # noqa:E402; attributed official copied kernel


def response(y: torch.Tensor) -> torch.Tensor:
    return y*torch.rsqrt(1+1.2*y.square())


def defects(y: torch.Tensor, matrix: torch.Tensor) -> dict[str,torch.Tensor]:
    a = 1.2
    u = 1+a*y.square()
    q0,q1 = y*u.rsqrt(),u.pow(-1.5)
    q2 = -3*a*y*u.pow(-2.5)
    q3 = (12*a*a*y.square()-3*a)*u.pow(-3.5)
    q4 = (45*a*a*y-60*a**3*y.pow(3))*u.pow(-4.5)
    def j(v):return F.linear(q1*v,matrix)
    def h(v,w):return F.linear(q2*v*w,matrix)
    def t(v,w,z):return F.linear(q3*v*w*z,matrix)
    f = F.linear(q0,matrix)
    jf = j(f);jjf = j(jf);hff = h(f,f)
    midpoint = -hff/24-jjf/6
    rk4 = (-j(j(jjf))/120+j(j(hff))/480-j(h(f,jf))/240-
           j(t(f,f,f))/720+h(f,jjf)/120-h(f,hff)/480+
           h(jf,jf)/160+t(f,f,jf)/480+F.linear(q4*f.pow(4),matrix)/2880)
    return dict(midpoint=midpoint,rk4=rk4)


def forcing(y: torch.Tensor, matrix: torch.Tensor,
            native: str, crossed: str) -> torch.Tensor:
    assert native in ('midpoint','rk4') and crossed in ('midpoint','rk4')
    if native == crossed:
        return torch.zeros_like(y)
    value = defects(y,matrix)
    prefactors = dict(midpoint=(1/4)**2,rk4=(1/2)**4)
    return prefactors[crossed]*value[crossed]-prefactors[native]*value[native]


def variation_solve(y: torch.Tensor, z: torch.Tensor, matrix: torch.Tensor,
                    steps: int, native: str, crossed: str) -> tuple[torch.Tensor,torch.Tensor]:
    assert steps > 0
    width = y.shape[1]
    state = torch.cat((y,z),dim=1)
    dt = 1/steps
    def drift(time,state,**unused):
        point,tangent = (part.contiguous() for part in state.split(width,dim=1))
        primal = F.linear(response(point),matrix)
        derivative = F.linear((1+1.2*point.square()).pow(-1.5)*tangent,matrix)
        derivative = derivative+forcing(point,matrix,native,crossed)
        return torch.cat((primal,derivative),dim=1)
    for index in range(steps):
        state = state+rk4_step_func(drift,index*dt,dt,(index+1)*dt,state)
    return tuple(part.contiguous() for part in state.split(width,dim=1))


def forced_solve(y: torch.Tensor, matrix: torch.Tensor, steps: int,
                 epsilon: torch.Tensor, native: str, crossed: str) -> torch.Tensor:
    """Separate full drift injection, for an independent JVP/FD construction."""
    dt = 1/steps
    def drift(time,point,**unused):
        return F.linear(response(point),matrix)+epsilon*forcing(point,matrix,native,crossed)
    for index in range(steps):
        y = y+rk4_step_func(drift,index*dt,dt,(index+1)*dt,y)
    return y


@torch.no_grad()
def model_response(model: torch.nn.Module, x: torch.Tensor, time: torch.Tensor,
                   steps: int, native: str, crossed: str) -> tuple[torch.Tensor,torch.Tensor]:
    """Inject local variational solves, keeping the original model computation."""
    layers = list(model.ode_layers())
    assert len(layers) == 6
    with forward_ad.dual_level(), ExitStack() as stack:
        for _,layer in layers:
            def solve(value,unused_steps,module=layer):
                point,tangent = forward_ad.unpack_dual(value)
                if tangent is None:
                    tangent = torch.zeros_like(point)
                point,tangent = variation_solve(point,tangent,module.operator(),
                                                steps,native,crossed)
                return forward_ad.make_dual(point,tangent)
            stack.enter_context(patch.object(layer,'solve',new=solve))
        dual = forward_ad.make_dual(x,torch.zeros_like(x))
        primal,tangent = forward_ad.unpack_dual(model(dual,time))
        assert tangent is not None
        return primal.clone(),tangent.clone()


def forced_model(model: torch.nn.Module, x: torch.Tensor, time: torch.Tensor,
                 steps: int, epsilon: torch.Tensor, native: str,
                 crossed: str) -> torch.Tensor:
    """Execute the original full model with f+epsilon*G in every local solve."""
    with ExitStack() as stack:
        layers = list(model.ode_layers())
        assert len(layers) == 6
        for _,layer in layers:
            def solve(value,unused_steps,module=layer):
                return forced_solve(value,module.operator(),steps,epsilon,native,crossed)
            stack.enter_context(patch.object(layer,'solve',new=solve))
        return model(x,time)
