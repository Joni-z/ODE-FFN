"""Literal nonlinear Heun flows and scalar updates independent of fitted models."""
import numpy as np


def scalar_heun(head,initial,bias,steps=32):
    value=float(initial);h=1/steps
    def field(z,t):
        denominator=max(1-t,.05)
        if head=='direct':return bias
        if head=='clean':return (bias-z)/denominator
        return bias-z/denominator
    for index in range(steps):
        first=field(value,index*h);predicted=value+h*first
        value+=h*(first+field(predicted,(index+1)*h))/2
    return value


def verify(null_projector,coefficients):
    rng=np.random.default_rng(0);d,w=6,4
    weight=rng.normal(size=(d,w));gamma=rng.normal(size=w);beta=rng.normal(size=w);bias=rng.normal(size=d)
    centre=np.eye(w)-np.ones((w,w))/w
    full_rank_controls=[]
    for ambient,width in ((2,4),(6,8)):
        matrix=rng.normal(size=(ambient,width))@(np.eye(width)-np.ones((width,width))/width)
        projector,rank,_=null_projector(matrix.T)
        assert rank==ambient and np.linalg.norm(projector)==0
        full_rank_controls.append(dict(dim=ambient,width=width,rank=rank))
    largest=0.;coefficient_errors=[]
    for zero_gamma,zero_bias in ((False,False),(True,False),(False,True),(True,True)):
        scale=np.zeros_like(gamma) if zero_gamma else gamma
        affine_beta=np.zeros_like(beta) if zero_bias else beta
        affine_bias=np.zeros_like(bias) if zero_bias else bias
        b=(weight*scale[None])@centre;c0=weight@affine_beta+affine_bias
        projector,rank,_=null_projector(b.T)
        assert rank==(0 if zero_gamma else 3)
        source=rng.normal(size=(17,d));hidden=rng.normal(size=(d,w))
        def raw(x,t):
            h=np.sin(x@hidden+.7*t)+.2*np.cos(x@hidden*1.3-t)
            centered=h-h.mean(1,keepdims=True)
            z=centered/np.sqrt(np.mean(centered**2,axis=1,keepdims=True)+1e-5)
            return (z*scale+affine_beta)@weight.T+affine_bias
        for head in ('clean','shifted','direct'):
            r,drift=coefficients(head)
            expected_r=scalar_heun(head,1.,0.);expected_d=scalar_heun(head,0.,1.)
            np.testing.assert_allclose([r,drift],[expected_r,expected_d],rtol=1e-13,atol=1e-13)
            coefficient_errors.append(float(max(abs(r-expected_r),abs(drift-expected_d))))
            value=source.copy();h=1/32
            def field(x,t):
                out=raw(x,t);delta=max(1-t,.05)
                if head=='clean':return (out-x)/delta
                if head=='shifted':return out-x/delta
                return out
            for index in range(32):
                first=field(value,index*h);predictor=value+h*first
                value+=h*(first+field(predictor,(index+1)*h))/2
            expected=r*(source@projector)+drift*(c0@projector)
            error=float(np.max(np.abs(value@projector-expected)));largest=max(largest,error)
            assert error<=1e-11
    r,_=coefficients('clean');rs,_=coefficients('shifted');assert r==rs
    return dict(status='PASS',literal_nonlinear_sample_identity_max=largest,
        coefficient_scalar_update_max=max(coefficient_errors),rank_zero_case_pass=True,zero_bias_case_pass=True,
        full_row_rank_controls=full_rank_controls,Heun32_skip_standard_deviation=r,
        continuous_skip_standard_deviation=.05/np.e,
        coefficients={head:dict(zip(('R','d'),coefficients(head))) for head in ('clean','shifted','direct')})
