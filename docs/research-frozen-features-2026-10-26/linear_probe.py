"""Fixed-cutoff linear probes; no trainable network parameters."""
import numpy as np

SITES=('input','pre_norm','post_norm')
LABELS=('mean','pair')
SCOPES=('shared','separate')
CUTOFF=1e-10

def design(h):return np.column_stack([np.ones(len(h)),h])

def solve(x,y):
    u,s,vh=np.linalg.svd(x,full_matrices=False)
    keep=s>CUTOFF*s[0]
    c=(vh[keep].T/s[keep])@(u[:,keep].T@y)
    residual=x@c-y
    normal=np.linalg.norm(x.T@residual)/(max(np.linalg.norm(x)*np.linalg.norm(y),1e-30))
    return c,dict(rank=int(keep.sum()),singular=s.tolist(),coefficient_frobenius=float(np.linalg.norm(c)),
        relative_normal_residual=float(normal),training_mse=(residual**2).mean(0).reshape(2,-1).mean(1).tolist())

def fit_all(features,targets):
    n=targets.shape[1];mid=n//2;coefficients={};records=[]
    for site in SITES:
        for scope in SCOPES:
            cs=[]
            for time_index in ([None] if scope=='shared' else range(len(targets))):
                folds=[]
                for fold,sl in enumerate([slice(0,mid),slice(mid,n)]):
                    if time_index is None:
                        x=design(features[site][:,sl].reshape(-1,features[site].shape[-1]))
                        y=targets[:,sl].reshape(-1,targets.shape[-1])
                    else:x=design(features[site][time_index,sl]);y=targets[time_index,sl]
                    c,record=solve(x,y);folds.append(c)
                    records.append(dict(site=site,scope=scope,time_index=time_index,training_fold=fold,**record))
                cs.append(np.array(folds))
            coefficients[site+'_'+scope]=cs[0] if scope=='shared' else np.array(cs)
    return coefficients,records

def predict(h,coefficients,rotation):
    n=len(h);mid=n//2;values=np.empty((n,32))
    for train,test in [(1,slice(0,mid)),(0,slice(mid,n))]:values[test]=design(h[test])@coefficients[train]
    return np.stack([values[:,:16]@rotation,values[:,16:]@rotation],axis=1)

def self_test():
    q=np.array([-2.,-1.,-.5,.5,1.,2.])
    rich=np.column_stack([q,np.tanh(q),2*q+3*np.tanh(q)])
    poor=np.column_stack([q,2*q,3*q])
    rich=np.tile(rich,(2,1));poor=np.tile(poor,(2,1))
    y=np.tile(np.column_stack([np.tanh(q),2*np.tanh(q)]),(2,1))
    # Two label groups with one output each for the algebra witness.
    c,info=solve(design(rich[:6]),y[:6]);assert info['rank']==3
    np.testing.assert_allclose(design(rich[6:])@c,y[6:],atol=1e-12)
    c,_=solve(design(poor[:6]),y[:6]);assert np.mean((design(poor[6:])@c-y[6:])**2)>.01
    hs=np.stack([rich,rich,rich]);ys=np.stack([y,2*y,3*y]);joined=design(hs[:,:6].reshape(-1,3))
    c,_=solve(joined,ys[:,:6].reshape(-1,2));joint_error=np.mean((joined@c-ys[:,:6].reshape(-1,2))**2)
    separate_error=0.
    for a,b in zip(hs,ys):
        c,_=solve(design(a[:6]),b[:6]);separate_error+=np.mean((design(a[6:])@c-b[6:])**2)
    assert joint_error>.01 and separate_error<1e-20
    rng=np.random.default_rng(0);rot,_=np.linalg.qr(rng.normal(size=(2,2)))
    c,_=solve(design(rich),y);other,_=solve(design(rich),y@rot)
    np.testing.assert_allclose(design(rich)@other,(design(rich)@c)@rot,atol=1e-12)
    # A fold's own targets never enter its fitted coefficients.
    x=design(rich);c0,_=solve(x[6:],y[6:]);changed=y.copy();changed[:6]+=10
    c1,_=solve(x[6:],changed[6:]);np.testing.assert_array_equal(c0,c1)
    return dict(status='PASS',checks=['available nonlinear feature','missing nonlinear feature',
        'structural rank deficiency','temporal sharing witness','output rotation','fold isolation'])
