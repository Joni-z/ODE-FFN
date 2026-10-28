"""Post-outcome accounting for the largest absolute component-time discrepancy."""
import argparse,csv,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch


def main(root):
    target=root/'tolerance-witness.json';assert not target.exists()
    spec=importlib.util.spec_from_file_location('frozen_posterior_identities',root/'source/check_identities.py')
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    row=max(csv.DictReader((root/'results/FM-coordinates.csv').open()),key=lambda v:float(v['complete_component_time_max']))
    p=next(v for v in f.priors() if v['name']==row['prior']);t=float(row['time']);z=f.grids(p['mu'].shape[1])[int(row['state'])]
    y=z/t;lam=t*t/(1-t)**2;ldot=2*t/(1-t)**3;v=f.state(p,y,lam);zt=torch.tensor(z);tt=torch.tensor(t)
    def packed(clock):
        a=f.spectral(p,zt/clock,clock*clock/(1-clock)**2)
        return torch.cat([a[k].reshape(-1) for k in ('b','s','ell','w','m')])
    actual=torch.autograd.functional.jacobian(packed,tt,vectorize=True).detach().numpy()
    residual=y-v['b'];ydot=-y/t
    db=ldot*np.einsum('kij,kj->ki',v['s'],residual)+lam*np.einsum('kij,j->ki',v['s'],ydot)
    ds=-ldot*(v['s']@v['s']);dl=ldot*(-.5*(np.trace(v['s'],axis1=1,axis2=2)+np.sum(residual*residual,axis=1)))-lam*(residual@ydot)
    dw=v['w']*(dl-v['w']@dl);dm=ldot*(v['c']@(y-v['m'])-.5*v['kappa'])-(lam/t)*v['c']@y
    expected=np.concatenate([a.reshape(-1) for a in (db,ds,dl,dw,dm)])
    difference=np.abs(actual-expected);bounds=f.TOL+f.TOL*np.abs(expected);index=int(np.argmax(difference))
    assert difference[index]==float(row['complete_component_time_max'])
    assert np.all(difference<=bounds)
    result=dict(status='PASS',post_outcome_arithmetic_accounting=True,original_checks_rerun=False,prior=p['name'],time=t,state=int(row['state']),
        vector_order=['component means','component covariances','unnormalized logits','normalized weights','global mean'],
        worst_absolute_error_index=index,actual_at_worst=float(actual[index]),expected_at_worst=float(expected[index]),
        absolute_error_at_worst=float(difference[index]),frozen_atol=f.TOL,frozen_rtol=f.TOL,allowed_error_at_worst=float(bounds[index]),
        maximum_fraction_of_allowed_error=float(np.max(difference/bounds)),
        actual_vector=actual.tolist(),expected_vector=expected.tolist(),tolerances_changed=False,new_quality_result=False)
    f.write(target,result)
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('_vector')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);main(parser.parse_args().root.resolve())
