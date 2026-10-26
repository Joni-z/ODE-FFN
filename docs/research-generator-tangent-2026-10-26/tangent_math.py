"""Exact derivative routes of the professor's scalar K2 polynomial."""
import numpy as np
import torch

def routes(x,u,a,p):
    a=np.asarray(a);px=x@p.T;up=u@p
    one=(u*a).T@x
    two=.5*(p.T@((u*a*a).T@x)+(u*a*a).T@px)
    xx=np.sum(x*x,1);uu=np.sum(u*u,1);xp=np.sum(x*px,1);upcross=np.sum(u*up,1);aa=a[:,0]
    e1=np.sum(aa**2*uu*xx)
    e2=np.sum(.25*aa**4*(np.sum(up*up,1)*xx+uu*np.sum(px*px,1)+2*upcross*xp))
    cross=np.sum(.5*aa**3*(upcross*xx+uu*xp))
    return one,two,np.array([e1,e2,cross])

def ratios(one,two,energy):
    den=float(energy[0]+energy[1]);coherent=float(np.sum(one*one)+np.sum(two*two))
    return dict(local_ratio=float((energy[0]+energy[1]+2*energy[2])/den) if den else None,
        coherent_ratio=float(np.sum((one+two)**2)/coherent) if coherent else None,
        route1_local_energy=float(energy[0]),route2_local_energy=float(energy[1]),local_cross=float(energy[2]),
        route1_gradient_norm=float(np.linalg.norm(one)),route2_gradient_norm=float(np.linalg.norm(two)),
        complete_gradient_norm=float(np.linalg.norm(one+two)))

def self_test():
    rng=np.random.default_rng(0);x=rng.normal(size=(7,4));u=rng.normal(size=(7,4));a=rng.uniform(.2,1,(7,1));p=rng.normal(size=(4,4))
    one,two,energy=routes(x,u,a,p)
    xx=torch.tensor(x);uu=torch.tensor(u);aa=torch.tensor(a);pp=torch.tensor(p,requires_grad=True)
    y=xx+aa*(xx@pp.T)+aa*aa/2*((xx@pp.T)@pp.T)
    native=torch.autograd.grad((uu*y).sum(),pp)[0].numpy()
    np.testing.assert_allclose(one+two,native,atol=1e-13)
    m1=a[:,:,None]*u[:,:,None]*x[:,None,:]
    m2=a[:,:,None]**2/2*((u@p)[:,:,None]*x[:,None,:]+u[:,:,None]*(x@p.T)[:,None,:])
    np.testing.assert_allclose(energy,[np.sum(m1*m1),np.sum(m2*m2),np.sum(m1*m2)],atol=1e-12)
    h=rng.normal(size=(4,4));at=.5;stationary=-np.eye(4)/at
    differential=at*h+.5*at*at*(stationary@h+h@stationary)
    np.testing.assert_array_equal(differential,np.zeros((4,4)))
    np.testing.assert_allclose(np.eye(4)+at*stationary+.5*at*at*stationary@stationary,.5*np.eye(4),atol=1e-14)
    g1,g2,e=routes(x,u,np.full((7,1),at),stationary);np.testing.assert_allclose(g1+g2,0,atol=1e-14)
    assert abs(ratios(g1,g2,e)['local_ratio'])<1e-14
    positive=.2*np.eye(4);g1,g2,e=routes(x,u,np.full((7,1),at),positive);assert ratios(g1,g2,e)['local_ratio']>1
    # Include the clock derivative when differentiating the input.
    z=torch.tensor(x[0],requires_grad=True);w=torch.tensor(rng.normal(size=4),requires_grad=True);pm=torch.tensor(p)
    clock=.2+.8*torch.sigmoid(torch.dot(w,z)/10);pz=pm@z;ppz=pm@pz
    y=z+clock*pz+clock*clock/2*ppz
    gradclock=torch.autograd.grad(clock,z,retain_graph=True)[0]
    expected=torch.eye(4,dtype=z.dtype)+clock*pm+clock*clock/2*(pm@pm)+(pz+clock*ppz)[:,None]*gradclock[None,:]
    actual=torch.stack([torch.autograd.grad(v,z,retain_graph=True)[0] for v in y]);torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    nativew=torch.autograd.grad((y*torch.tensor(u[0])).sum(),w)[0]
    probability=torch.sigmoid(torch.dot(w,z)/10)
    predictedw=torch.dot(torch.tensor(u[0]),pz+clock*ppz)*(.08*probability*(1-probability))*z
    torch.testing.assert_close(nativew,predictedw,atol=1e-12,rtol=1e-12)
    # Orthogonality/envelope identity for a refitted raw-linear projection.
    design=torch.tensor(rng.normal(size=(19,3)));target=torch.tanh(design);theta=torch.tensor(.7,requires_grad=True)
    field=torch.sin(theta*design);coef=torch.linalg.solve(design.T@design,design.T@(field-target));residual=field-target-design@coef
    loss=residual.square().mean();g=torch.autograd.grad(loss,theta,retain_graph=True)[0]
    envelope=2*(field*residual.detach()).sum()/field.numel();g2=torch.autograd.grad(envelope,theta)[0]
    torch.testing.assert_close(g,g2,atol=1e-13,rtol=1e-13)
    return dict(status='PASS',checks=['complete P derivative','per-row route energies','stationary nonzero map','constructive witness','full input clock derivative','controller parameter derivative','projection envelope derivative'])
