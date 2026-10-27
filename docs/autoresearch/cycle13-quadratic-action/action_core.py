"""Finite K2 action geometry; clocks in the envelopes are relaxations."""
from __future__ import annotations

import numpy as np

KEYS = ('increment_energy','quadratic_energy','second_action_energy','second_orthogonal_energy',
        'sameclock_error_energy','same_free_error_energy','same_box_error_energy',
        'center_free_error_energy','center_box_error_energy','same_free_clock','same_box_clock',
        'center_free_clock','center_box_clock','clock')


def rotate(x, g, transpose=False):
    if g is None: return x
    if transpose: g=-g
    norm=np.sum(g*g,axis=1,keepdims=True);product=np.sum(x[:,1:]*g,axis=1,keepdims=True)
    return x+np.concatenate((-2*(norm*x[:,:1]+product)/(1+norm),
                              2*g*(x[:,:1]-product)/(1+norm)),axis=1)


def norm2(x): return np.einsum('nd,nd->n',x,x)


def projection(d,v):
    denominator=norm2(v)
    r=np.divide(np.einsum('nd,nd->n',d,v),denominator,
                out=np.zeros(len(d)),where=denominator!=0)
    box=np.clip(r,.2,1.)
    return r,box,norm2(d-r[:,None]*v),norm2(d-box[:,None]*v)


def actions(p,x,s,g=None):
    z=rotate(x,g,True);u=z@p.T;w=u@p.T
    d=s*u+.5*s*s*w;v=u+.3*w
    same,box,same_error,box_error=projection(d,u)
    center,cbox,center_error,cbox_error=projection(d,v)
    residual=.5*s*(s-.6)*w
    np.testing.assert_allclose(d-s*v,residual,atol=1e-10,rtol=1e-10)
    _,_,orthogonal,_=projection(w,u)
    samples=np.column_stack((norm2(d),norm2(.5*s*s*w),norm2(w),orthogonal,norm2(residual),
        same_error,box_error,center_error,cbox_error,same,box,center,cbox,s[:,0]))
    assert np.isfinite(samples).all()
    assert np.all(center_error<=norm2(residual)+1e-10)
    assert np.all(cbox_error<=norm2(residual)+1e-10)
    return samples,x+rotate(d,g)


def summarize(samples):
    values={key:samples[:,i] for i,key in enumerate(KEYS)}
    energy=values['increment_energy'].mean()
    result=dict(count=len(samples),increment_rms=float(np.sqrt(energy/64)),
                clock_mean=float(values['clock'].mean()),clock_std=float(values['clock'].std()))
    for name in ('quadratic','sameclock_error','same_free_error','same_box_error',
                 'center_free_error','center_box_error'):
        result[name+'_relative_rms']=float(np.sqrt(values[name+'_energy'].mean()/energy)) if energy>0 else None
    total=values['second_action_energy'].mean()
    result['second_action_orthogonal_fraction']=float(values['second_orthogonal_energy'].mean()/total) if total>0 else None
    for prefix in ('same','center'):
        free=values[prefix+'_free_clock']
        result[prefix+'_clipped_fraction']=float(np.mean((free<.2)|(free>1)))
        result[prefix+'_free_clock_min']=float(free.min());result[prefix+'_free_clock_max']=float(free.max())
    value=result['center_box_error_relative_rms']
    result['center_box_below_one_percent']=value<=.01 if value is not None else None
    return result
