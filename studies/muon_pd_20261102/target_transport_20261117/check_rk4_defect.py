"""Independent high-precision scalar and matrix controls; no learned weights."""
import argparse
from decimal import Decimal, localcontext
import json
from pathlib import Path

import numpy as np
from rk4_defect import local_defects,response_jet,explicit_defects


def scalar_control() -> dict:
    rows=[]
    prediction=local_defects(np.array([[.7]]),np.array([[-.4]]))
    with localcontext() as context:
        context.prec=80
        a,p,y0=map(Decimal,['1.2','-.4','.7'])
        def q(y):return p*y/(1+a*y*y).sqrt()
        def primitive(y):
            r=(1+a*y*y).sqrt()
            return r+(y/(1+r)).ln()
        for index in range(7):
            h=Decimal('.1')/(2**index)
            endpoint=y0
            for _ in range(100):
                residual=primitive(endpoint)-primitive(y0)-p*h
                correction=residual*endpoint/(1+a*endpoint*endpoint).sqrt()
                endpoint-=correction
                if abs(correction)<Decimal('1e-70'):break
            assert abs(primitive(endpoint)-primitive(y0)-p*h)<Decimal('1e-69')
            k1=q(y0);k2=q(y0+h*k1/2);k3=q(y0+h*k2/2);k4=q(y0+h*k3)
            rk4=y0+h*(k1+2*k2+2*k3+k4)/6
            midpoint=y0+h*q(y0+h*k1/2)
            for method,value,order in [('midpoint',midpoint,3),('rk4',rk4,5)]:
                actual=float((value-endpoint)/h**order)
                predicted=float(prediction[method][0,0])
                rows.append(dict(method=method,step=float(h),actual_scaled_defect=actual,
                    predicted=predicted,relative_error=abs(actual-predicted)/abs(predicted)))
    for method in ('midpoint','rk4'):
        errors=[r['relative_error'] for r in rows if r['method']==method]
        assert errors[-1]<.02 and all(b<a for a,b in zip(errors[-4:-1],errors[-3:]))
    return dict(passed=True,decimal_precision=80,rows=rows)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    rng=np.random.default_rng(0)
    y=rng.normal(size=(16,5));p=rng.normal(size=(5,5))*.2
    linear=local_defects(y,p,linear=True)
    expected=-y@np.linalg.matrix_power(p,5).T/120
    error=float(np.max(np.abs(linear['rk4']-expected)))
    assert error<1e-12
    nonlinear=local_defects(y,p)
    jet=response_jet(y);f=jet[0]@p.T
    j_f=(jet[1]*f)@p.T
    repeated=(jet[1]*j_f)@p.T
    curvature=(jet[2]*f*f)@p.T
    mid=-curvature/24-repeated/6
    mid_error=float(np.max(np.abs(mid-nonlinear['midpoint'])))
    assert mid_error<1e-12
    explicit=explicit_defects(y,p)
    explicit_error={method:float(np.max(np.abs(explicit[method]-nonlinear[method])))
                    for method in ('midpoint','rk4')}
    assert all(error<1e-12 for error in explicit_error.values())
    zero=local_defects(y,np.zeros_like(p))
    assert not np.any(zero['rk4']) and not np.any(zero['midpoint'])
    result=dict(passed=True,seed=0,linear_rk4_error=error,midpoint_formula_error=mid_error,
                explicit_formula_errors=explicit_error,scalar=scalar_control(),learned_checkpoints_used=0)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);assert not output.exists()
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,linear_rk4_error=error,midpoint_formula_error=mid_error)))


if __name__=='__main__':main()
