"""Invariant-conditioned rotations and explicit same-budget controls."""
import torch
from torch import nn
from torch.nn import functional as F


class InvariantClockLayer(nn.Module):
    def __init__(self,width=64,hidden=128,kind='rotation'):
        super().__init__()
        if width%2 or kind not in ('rotation','euler','normalized_euler'):raise ValueError((width,kind))
        self.width,self.kind=width,kind
        self.clock=nn.Sequential(nn.Linear(width//2,hidden),nn.SiLU(),nn.Linear(hidden,width//2))
        self.frequency=nn.Parameter(torch.zeros(width//2))

    @staticmethod
    def invariants(x):return .5*x.reshape(*x.shape[:-1],-1,2).square().sum(-1)

    def clocks(self,actions):return .2+.8*torch.sigmoid(self.clock(actions)/10.)

    def rates(self,x):return self.frequency*self.clocks(self.invariants(x))

    def vector_field(self,x):
        pairs=x.reshape(*x.shape[:-1],-1,2);a,b=pairs.unbind(-1)
        return (self.rates(x).unsqueeze(-1)*torch.stack((-b,a),-1)).reshape_as(x)

    def forward(self,x,duration=1.):
        pairs=x.reshape(*x.shape[:-1],-1,2);a,b=pairs.unbind(-1);angle=duration*self.rates(x)
        if self.kind=='rotation':
            c,s=torch.cos(angle),torch.sin(angle);out=torch.stack((c*a-s*b,s*a+c*b),-1)
        else:
            out=torch.stack((a-angle*b,b+angle*a),-1)
            if self.kind=='normalized_euler':out=out*torch.rsqrt(1+angle.square()).unsqueeze(-1)
        return out.reshape_as(x)

    def generator(self):
        p=self.frequency.new_zeros(self.width,self.width);index=torch.arange(self.width//2,device=p.device)
        p[2*index,2*index+1]=-self.frequency;p[2*index+1,2*index]=self.frequency
        return p

    def rate_derivative(self,x):
        """Literal clock derivative, independently of a full-map AD Jacobian."""
        assert x.ndim==1
        actions=self.invariants(x);first,last=self.clock[0],self.clock[2]
        z=F.linear(actions,first.weight,first.bias);sig=torch.sigmoid(z)
        middle=F.silu(z);top=torch.sigmoid(F.linear(middle,last.weight,last.bias)/10.)
        derivative=sig+z*sig*(1-sig)
        return ((self.frequency*.08*top*(1-top))[:,None]*last.weight*derivative[None,:])@first.weight

    def complete_jacobian(self,x,duration=1.):
        assert x.ndim==1
        d=self.width;m=d//2;pairs=x.reshape(m,2);a,b=pairs.unbind(-1)
        u=x.new_zeros(d,m);v=x.new_zeros(m,d);index=torch.arange(m,device=x.device)
        u[2*index,index]=-b;u[2*index+1,index]=a;v[index,2*index]=a;v[index,2*index+1]=b
        angle=duration*self.rates(x);feedback=u@self.rate_derivative(x)@v
        if self.kind=='euler':
            linear=x.new_zeros(d,d);linear[2*index,2*index+1]=-angle;linear[2*index+1,2*index]=angle
            return torch.eye(d,device=x.device,dtype=x.dtype)+linear+duration*feedback
        if self.kind=='normalized_euler':
            feedback=u@(self.rate_derivative(x)/(1+angle.square())[:,None])@v;angle=torch.atan(angle)
        c,s=torch.cos(angle),torch.sin(angle);rotation=x.new_zeros(d,d)
        rotation[2*index,2*index]=c;rotation[2*index,2*index+1]=-s
        rotation[2*index+1,2*index]=s;rotation[2*index+1,2*index+1]=c
        return rotation@(torch.eye(d,device=x.device,dtype=x.dtype)+duration*feedback)


class InvariantClockFFN(nn.Module):
    def __init__(self,width=64,hidden=128,kind='rotation'):
        super().__init__();self.ode1=InvariantClockLayer(width,hidden,kind);self.act=nn.SiLU();self.ode2=InvariantClockLayer(width,hidden,kind)

    def forward(self,x):return self.ode2(self.act(self.ode1(x)))


class IdentitySiLU(nn.Module):
    """Same initial function, with ordinary or shifted-matrix coordinates."""
    def __init__(self,width=64,kind='silu_identity'):
        super().__init__();self.kind=kind
        if kind not in ('silu_identity','fixed06_identity'):raise ValueError(kind)
        initial=torch.eye(width) if kind=='silu_identity' else torch.zeros(width,width)
        self.matrix1=nn.Parameter(initial.clone());self.matrix2=nn.Parameter(initial.clone())

    def forward(self,x):
        if self.kind=='silu_identity':return F.linear(F.silu(F.linear(x,self.matrix1)),self.matrix2)
        hidden=F.silu(x+.6*F.linear(x,self.matrix1))
        return hidden+.6*F.linear(hidden,self.matrix2)
