"""Fixed-cutoff affine residual probes and mathematical witnesses."""
import itertools
import numpy as np

def design(h):return np.column_stack([np.ones(len(h)),h])

def solve(x,y):
    u,s,vh=np.linalg.svd(x,full_matrices=False);keep=s>s[0]*1e-10
    c=(vh[keep].T/s[keep])@(u[:,keep].T@y);residual=x@c-y
    return c,dict(rank=int(keep.sum()),singular=s.tolist(),coefficient_frobenius=float(np.linalg.norm(c)),
        relative_normal_residual=float(np.linalg.norm(x.T@residual)/max(np.linalg.norm(x)*np.linalg.norm(y),1e-30)),
        training_mse=(residual**2).mean(0).reshape(2,-1).mean(1).tolist())

def predict(h,c,rotation):
    n=len(h);mid=n//2;physical=np.empty((n,32))
    for train,sl in [(1,slice(0,mid)),(0,slice(mid,n))]:physical[sl]=design(h[sl])@c[train]
    return np.stack([physical[:,:16]@rotation,physical[:,16:]@rotation],axis=1)

def self_test():
    points=np.array(list(itertools.product([-1.,1.],[-2.,-.5,.5,2.])))
    silu=lambda x:x/(1+np.exp(-x))
    hidden=np.column_stack([silu(points[:,0]),silu(-points[:,0]),silu(points[:,1]),silu(-points[:,1])])
    logit=hidden[:,0]-hidden[:,1]
    np.testing.assert_allclose(logit,points[:,0],atol=1e-14)
    duration=.2+.8/(1+np.exp(-logit/10))
    exported=points.copy();exported[:,1]*=1+duration+.5*duration**2
    target=np.column_stack([silu(points[:,1]),2*silu(points[:,1])])
    full,info=solve(design(hidden),target);poor,_=solve(design(exported),target)
    np.testing.assert_allclose(design(hidden)@full,target,atol=1e-12)
    assert np.mean((design(exported)@poor-target)**2)>.01
    native=np.column_stack([points[:,1],points[:,1]])
    final_features=np.column_stack([points[:,1],points[:,1]**2,points[:,0]])
    x=design(final_features);direct,_=solve(x,target);residual,_=solve(x,target-native)
    np.testing.assert_allclose(x@direct,native+x@residual,atol=1e-12)
    h=np.array([.3,-.4,.8]);p=np.array([[.2,.4,-.1],[-.3,.1,.5],[.4,-.2,.3]]);w=np.array([.7,-.2,.9])
    def layer(x):
        a=.2+.8/(1+np.exp(-(w@x+.1)/10))
        return x+a*p@x+.5*a*a*p@p@x
    prob=1/(1+np.exp(-(w@h+.1)/10));a=.2+.8*prob;gradient=.08*prob*(1-prob)*w
    clock=np.outer(p@h+a*p@p@h,gradient)
    jac=np.eye(3)+a*p+.5*a*a*p@p+clock
    assert np.linalg.svd(clock,compute_uv=False)[1]<1e-14
    for direction in np.eye(3):
        actual=(layer(h+1e-6*direction)-layer(h-1e-6*direction))/2e-6
        np.testing.assert_allclose(jac@direction,actual,atol=2e-9,rtol=2e-9)
    # Rank-deficient minimum-norm coefficients still define a valid function.
    assert info['rank']<design(hidden).shape[1]
    changed=target.copy();changed[:4]+=10
    unchanged,_=solve(design(hidden[4:]),target[4:]);other,_=solve(design(hidden[4:]),changed[4:])
    np.testing.assert_array_equal(unchanged,other)
    return dict(status='PASS',checks=['nonlinear hidden/compression witness','residual/direct-head identity',
        'K2 directional Jacobian','rank-one clock response','rank deficiency','fold isolation'])
