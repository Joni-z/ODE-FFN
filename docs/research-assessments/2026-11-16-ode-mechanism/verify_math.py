"""Small deterministic algebra checks; no model training or research scores."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

P = np.array([[0.3, -0.8], [0.6, -0.2]], dtype=np.float64)
h = np.array([0.9, -0.7], dtype=np.float64)
r = 1.2

def d(x): return 1 / np.sqrt(1 + r * x*x)
def f(x): return P @ (d(x)*x)
def jf(x): return P @ np.diag(d(x)**3)
def frozen(x, s): return expm(s * P @ np.diag(d(x))) @ x
def exact(x, s):
    sol = solve_ivp(lambda t,y:f(y), (0,s), x, method='DOP853', rtol=1e-12, atol=1e-13)
    assert sol.success
    return sol.y[:,-1]
def midpoint(x,s,n):
    y=x.copy();dt=s/n
    for _ in range(n): y = y + dt * f(y + dt*f(y)/2)
    return y

def main():
    expected = P @ (np.diag(d(h)**3)-np.diag(d(h))) @ f(h)
    defects=[]
    for e in [0.04,0.02,0.01,0.005]:
        observed=(frozen(frozen(h,e),e)-frozen(h,2*e))/(e*e)
        defects.append({'step':e,'scaled_defect':observed.tolist(),
                        'error_to_leading_coefficient':float(np.linalg.norm(observed-expected))})
    assert defects[-1]['error_to_leading_coefficient'] < defects[0]['error_to_leading_coefficient']/6
    A=P@np.diag(d(h));same_generator_error=np.linalg.norm(expm(.2*A)@expm(.3*A)@h-expm(.5*A)@h)
    exact_composition_error=np.linalg.norm(exact(exact(h,.2),.3)-exact(h,.5))
    assert same_generator_error<1e-12 and exact_composition_error<1e-10
    ref=exact(h,1.0)
    refinement=[{'steps':n,'P_applications':2*n,'error':float(np.linalg.norm(midpoint(h,1.,n)-ref))} for n in [4,8,16,32]]
    assert all(refinement[i]['error']/refinement[i+1]['error']>3 for i in range(3))
    u=f(h);eps=1e-5
    directional=(f(h+eps*u)-f(h-eps*u))/(2*eps)
    assert np.linalg.norm(directional-jf(h)@u)<1e-9
    # Scalar conditional exponent can fail injectivity even though exp(A)>0.
    x=1.;p=3.;dd=float(d(x));derivative=np.exp(p*dd)*(1-p*r*x*x*dd**3)
    assert derivative<0 # derivative at zero is exp(3)>0
    # First-order transport: perturbed field f+epsilon*b, z'=J_f z+b.
    b=lambda t,x:np.array([np.sin(t)+.2*x[1],.3*np.cos(t)])
    def aug(t,w):
        y,z=w[:2],w[2:]
        return np.concatenate((f(y),jf(y)@z+b(t,y)))
    sol=solve_ivp(aug,(0,1),np.r_[h,[0.,0.]],method='DOP853',rtol=1e-12,atol=1e-13)
    sensitivity=sol.y[2:,-1];transport=[]
    for e in [.01,.005,.0025]:
        pert=solve_ivp(lambda t,y:f(y)+e*b(t,y),(0,1),h,method='DOP853',rtol=1e-12,atol=1e-13)
        residual=np.linalg.norm((pert.y[:,-1]-ref)/e-sensitivity)
        transport.append({'epsilon':e,'first_order_residual':float(residual)})
    assert transport[-1]['first_order_residual']<transport[0]['first_order_residual']/3
    skew=np.array([[0.,-1.],[1.,0.]])
    discrete_norm=float(np.linalg.norm(np.eye(2)+.1*skew,2))
    assert discrete_norm>1 and np.linalg.norm(expm(.1*skew),2)<1+1e-12
    result={'passed':True,'scope':'algebra verification, not learned-model evidence',
            'semigroup_leading_coefficient':expected.tolist(),'conditional_composition':defects,
            'same_generator_composition_error':float(same_generator_error),
            'true_flow_composition_error':float(exact_composition_error),
            'midpoint_refinement':refinement,'conditional_scalar_derivative':float(derivative),
            'transport_sensitivity':transport,
            'skew_euler_step_norm':discrete_norm,'skew_continuous_exp_norm':float(np.linalg.norm(expm(.1*skew),2))}
    out=Path(__file__).with_name('math-verification.json')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__': main()
