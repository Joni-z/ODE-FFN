"""Algebraic fixtures before any fitted scalar-envelope outcomes."""
import numpy as np
import torch

from action_core import actions,projection,rotate


def verify():
    rng=np.random.default_rng(1701);p=rng.normal(size=(5,5))*.2
    x=rng.normal(size=(7,5));s=rng.uniform(.2,1,size=(7,1));g=rng.normal(size=(7,4))*.3
    samples,y=actions(p,x,s,g)
    z=rotate(x,g,True);u=z@p.T;w=u@p.T;d=s*u+.5*s*s*w
    np.testing.assert_allclose(y,x+rotate(d,g),atol=1e-12,rtol=0)
    for v in (u,u+.3*w):
        free,box,_,error=projection(d,v)
        for i in range(len(x)):
            grid=np.linspace(.2,1,1001);grid_errors=np.sum((d[i]-grid[:,None]*v[i])**2,axis=1)
            assert error[i]<=grid_errors.min()+1e-12
            np.testing.assert_allclose(np.sum((d[i]-free[i]*v[i])*v[i]),0,atol=1e-12)
    _,_,freezero,boxzero=projection(np.ones((2,5)),np.zeros((2,5)))
    np.testing.assert_array_equal(freezero,np.full(2,5.));np.testing.assert_array_equal(boxzero,freezero)
    for multiplier,expected in ((.1,.2),(2.,1.)):
        _,box,_,_=projection(multiplier*np.ones((1,5)),np.ones((1,5)))
        np.testing.assert_allclose(box,expected,atol=1e-12)
    eye=np.eye(2);turn=np.array([[0.,-1.],[1.,0.]])
    witness=[]
    for matrix in (eye,turn):
        u=matrix@eye[0];w=matrix@u
        witness.append(float(np.sum((w-u*(u@w)/(u@u))**2)/(w@w)))
    np.testing.assert_allclose(witness,[0.,1.],atol=1e-12)
    np.testing.assert_array_equal(np.linalg.svd(eye,compute_uv=False),np.linalg.svd(turn,compute_uv=False))
    # Full state/parameter derivatives, including both live controls, use two independent forms.
    torch.set_default_dtype(torch.float64)
    tx=torch.tensor(x[:2],requires_grad=True);tp=torch.tensor(p,requires_grad=True)
    gate=torch.linspace(-.2,.3,5,requires_grad=True)
    frame=torch.linspace(-.1,.2,20).reshape(4,5).requires_grad_(True)
    def qmatrix(gate_value):
        one=torch.cat((torch.ones(1,dtype=tx.dtype),gate_value));h=torch.eye(5)-2*torch.outer(one,one)/(one@one)
        h0=torch.eye(5);h0[0,0]=-1
        return h@h0
    def direct():
        ys=[]
        for row in tx:
            clock=.2+.8*torch.sigmoid(row@gate);q=qmatrix(torch.tanh(frame@row))
            z=q.T@row;u=tp@z;w=tp@u
            ys.append(row+q@(clock*u+.5*clock**2*w))
        return torch.stack(ys)
    def centered():
        ys=[]
        for row in tx:
            clock=.2+.8*torch.sigmoid(row@gate);q=qmatrix(torch.tanh(frame@row))
            z=q.T@row
            ys.append(row+q@(clock*(tp+.3*tp@tp)@z+.5*clock*(clock-.6)*tp@tp@z))
        return torch.stack(ys)
    a,b=direct(),centered();torch.testing.assert_close(a,b,atol=1e-12,rtol=0)
    inputs=(tx,tp,gate,frame)
    left=torch.autograd.grad(a.square().sum(),inputs);right=torch.autograd.grad(b.square().sum(),inputs)
    maximum=max(float((l-r).abs().max()) for l,r in zip(left,right))
    assert maximum<=1e-10
    torch.set_default_dtype(torch.float32)
    return dict(status='PASS',live_frame_clock_AD_max=maximum,flat_singular_witness=witness,
                projection_grid_zero_boundary_checked=True,centered_identity=True)


if __name__=='__main__': print(verify())
