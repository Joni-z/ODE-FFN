"""Empirical FID through an exact low-rank trace identity."""
import numpy as np

def fid(x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    assert x.ndim==y.ndim==2 and x.shape[1]==y.shape[1] and min(len(x),len(y))>1
    assert np.isfinite(x).all() and np.isfinite(y).all()
    mx=x.mean(0);my=y.mean(0);a=(x-mx)/np.sqrt(len(x)-1);b=(y-my)/np.sqrt(len(y)-1)
    trace=float(np.linalg.svd(a@b.T,compute_uv=False).sum())
    result=float(np.sum((mx-my)**2)+np.sum(a*a)+np.sum(b*b)-2*trace)
    assert result>=-1e-6,(result,'negative beyond roundoff')
    return max(result,0.)

def uint8_images(x):
    a=np.asarray(x,dtype=np.float32);assert a.ndim==4 and a.shape[1:]==(3,32,32) and np.isfinite(a).all()
    return np.rint((np.clip(a,-1,1)+np.float32(1))*np.float32(127.5)).astype(np.uint8)
