"""A local scalar flow-map certificate; not a full-network training result."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.data import GaussianMixture


def coefficients(mu,variance,time):
    remaining=1-time
    marginal=remaining**2+time**2*variance
    return ((time*variance-remaining)/marginal,remaining*mu/marginal,
            time*mu/marginal,marginal)


def quotient(x,alpha,beta,kappa):
    x=np.asarray(x,dtype=np.float64)
    value=np.full_like(x,alpha+beta*kappa)
    active=x!=0
    value[active]=alpha+beta*np.tanh(kappa*x[active])/x[active]
    return value


def positive_root(alpha,beta,kappa):
    if alpha+beta*kappa<=0:return None
    assert alpha<0
    low,high=0.,max(1.,2*beta/(-alpha))
    for _ in range(80):
        middle=(low+high)/2
        if alpha+beta*math.tanh(kappa*middle)/middle>0:low=middle
        else:high=middle
    return (low+high)/2


def certify(manifest_path,output):
    manifest=json.loads(manifest_path.read_text())
    spec=manifest['datasets']['rotated_product_bimodal_16d']['distribution']
    locations,variances=np.array(spec['locations']),np.array(spec['variances'])
    delta,center=.001,.6
    rules={n:np.polynomial.legendre.leggauss(n) for n in (160,320)}
    rows=[];maximum_oracle_error=maximum_witness_error=maximum_quadrature_difference=0.
    selected=int(np.argmin(np.abs(locations-1.)))
    example=None
    for axis,(mu,variance) in enumerate(zip(locations,variances)):
        for time in (.1,.3,.5,.7,.9):
            alpha,beta,kappa,marginal=coefficients(mu,variance,time)
            assert alpha<0
            maximum=alpha+beta*kappa
            limit=time*mu+12*np.sqrt(marginal)
            root=positive_root(alpha,beta,kappa)
            if root is not None:assert root<limit
            def integrate(left,right,count):
                if right<=left:return 0.
                nodes,weights=rules[count]
                x=left+(nodes+1)*(right-left)/2
                velocity=alpha*x+beta*np.tanh(kappa*x)
                density=(np.exp(-.5*(x-time*mu)**2/marginal)+np.exp(-.5*(x+time*mu)**2/marginal))/(2*np.sqrt(2*np.pi*marginal))
                return float(delta**2*(right-left)*np.dot(weights,velocity**2*density))
            energies=[]
            for count in (160,320):
                expanding=integrate(0,root,count) if root is not None else 0.
                contracting=integrate(root or 0.,limit,count)
                energies.append((expanding,contracting))
            error=max(abs(a-b) for a,b in zip(*energies));maximum_quadrature_difference=max(maximum_quadrature_difference,error)
            assert error<1e-12
            expanding,contracting=energies[-1]
            x=np.linspace(-limit,limit,1001)
            q=quotient(x,alpha,beta,kappa)
            velocity=alpha*x+beta*np.tanh(kappa*x)
            distribution=GaussianMixture(np.array([[-mu],[mu]]),np.full((2,1),variance),np.ones(2)/2)
            oracle=distribution.velocity(x[:,None],time)[:,0]
            oracle_error=float(np.abs(velocity-oracle).max());maximum_oracle_error=max(maximum_oracle_error,oracle_error)
            np.testing.assert_allclose(velocity,oracle,atol=2e-12,rtol=2e-12)
            target=x+delta*velocity
            midpoint=delta*(alpha+beta*kappa/2)
            p=midpoint/center
            duration=center+delta*q-midpoint
            low,high=center-delta*beta*kappa/2,center+delta*beta*kappa/2
            assert .2<low<=high<1
            additive=x+center*p*x+(duration-center)*x
            rate_low,rate_high=math.log1p(delta*alpha),math.log1p(delta*maximum)
            rate_midpoint=(rate_low+rate_high)/2
            exact_duration=center+np.log1p(delta*q)-rate_midpoint
            exact=np.exp(rate_midpoint+exact_duration-center)*x
            assert .2<float(exact_duration.min())<=float(exact_duration.max())<1
            witness_error=max(float(np.abs(target-additive).max()),float(np.abs(target-exact).max()))
            maximum_witness_error=max(maximum_witness_error,witness_error)
            assert witness_error<1e-13
            row=dict(axis=axis,mu=float(mu),variance=float(variance),time=time,delta=delta,
                alpha=alpha,beta=beta,kappa=kappa,coefficient_min=alpha,coefficient_max=maximum,
                target_flow_jacobian_lower_bound=1+delta*alpha,positive_velocity_root=root,
                expanding_increment_energy=expanding,contracting_increment_energy=contracting,
                positive_clock_risk_floor=min(expanding,contracting),
                floor_over_target_increment_energy=min(expanding,contracting)/(expanding+contracting),
                euler_additive_P=p,euler_gate_min=low,euler_gate_max=high,
                exact_additive_P=rate_midpoint/center,
                exact_gate_min=center+(rate_low-rate_high)/2,
                exact_gate_max=center+(rate_high-rate_low)/2,
                quadrature_absolute_difference=error)
            rows.append(row)
            if axis==selected and time==.5:
                example=dict(row=row,x=x,velocity=velocity,oracle=oracle,target=target,
                    additive=additive,exact=exact,duration=duration,exact_duration=exact_duration)
    assert example is not None
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'example.npz',**{k:v for k,v in example.items() if k!='row'})
    result=dict(state='PASS',rows=rows,example=example['row'],delta=delta,
        maximum_oracle_difference=maximum_oracle_error,maximum_witness_difference=maximum_witness_error,
        maximum_quadrature_absolute_difference=maximum_quadrature_difference,
        quadrature='Gauss-Legendre160/320 on sign-separated intervals; symmetric marginal truncated at mode+12sigma. Numerical agreement is reported, not an interval-arithmetic certificate.',
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        example_sha256=hashlib.sha256((output/'example.npz').read_bytes()).hexdigest(),
        scope='Latent scalar single-flow-map families with arbitrary smooth positive controllers. Not a lower bound on the actual multilayer velocity network, and not a learned finite-MLP controller result.')
    (output/'certificate.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=certify(args.manifest,args.output)
    print(json.dumps({k:result[k] for k in ('state','example','maximum_oracle_difference','maximum_witness_difference','maximum_quadrature_absolute_difference')}))
