"""Exact six-product form of the same two-defect coupled variation equation."""
from functools import partial
import torch
import torch.nn.functional as F
from hidden_defect_transport import rk4_step_func


def drift(y,z,matrix,native,crossed,reuse_powers=False):
    assert native in ('midpoint','rk4') and crossed in ('midpoint','rk4')
    a=1.2;u=1+a*y.square();r=u.rsqrt();f=F.linear(y*r,matrix)
    if reuse_powers:
        inv=u.reciprocal();p3=r*inv;p5=p3*inv;p7=p5*inv;p9=p7*inv
    else:p3,p5,p7,p9=(u.pow(v) for v in (-1.5,-2.5,-3.5,-4.5))
    if native==crossed:return f,F.linear(p3*z,matrix)
    q1=p3;q2=-3*a*y*p5
    q3=(12*a*a*y.square()-3*a)*p7
    q4=(45*a*a*y-60*a**3*y.pow(3))*p9
    jf=F.linear(q1*f,matrix)
    jjf=F.linear(q1*jf,matrix)
    hff=F.linear(q2*f.square(),matrix)
    w=F.linear(q1*(-jjf/120+hff/480)-q2*f*jf/240-q3*f.pow(3)/720,matrix)
    rk=q1*w+q2*(f*jjf/120-f*hff/480+jf.square()/160)+q3*f.square()*jf/480+q4*f.pow(4)/2880
    mid=-q2*f.square()/24-q1*jf/6
    cr=(1/2)**4*(int(crossed=='rk4')-int(native=='rk4'))
    cm=(1/4)**2*(int(crossed=='midpoint')-int(native=='midpoint'))
    return f,F.linear(q1*z+cr*rk+cm*mid,matrix)


def coupled_step(state,matrix,dt,native,crossed,reuse_powers=False):
    width=matrix.shape[0]
    def field(time,value,**unused):
        y,z=(part.contiguous() for part in value.split(width,dim=1))
        return torch.cat(drift(y,z,matrix,native,crossed,reuse_powers),dim=1)
    return state+rk4_step_func(field,0.,dt,dt,state)


def make_compiled_step(reuse_powers=False):
    return torch.compile(partial(coupled_step,reuse_powers=reuse_powers),fullgraph=True,dynamic=False)
