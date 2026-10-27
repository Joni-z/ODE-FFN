"""Post-outcome descriptive decomposition; does not select architectures."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=args.root;out=args.output;out.mkdir(exist_ok=False)
sys.path.insert(0,str(root/'source'));import study
_,ref=study.setup(root);study.verify(root,ref);law=study.target(root)
with np.load(root/'banks/field.npz') as data:x,t,oracle=[data[k] for k in ('x','t','oracle')]
q=x.astype(np.float64)@law.rotation;d=(1-t.astype(np.float64))**2+.04*t.astype(np.float64)**2
kappa=t/d;coordinate=kappa*q*law.locations;m0=law.locations*np.tanh(coordinate)
logits=(kappa*q)@law.means.T;logits-=logits.max(-1,keepdims=True)
p=np.exp(logits);p/=p.sum(-1,keepdims=True);m1=p@law.means
_,_,w=law.oracle(x,t,True)
c0=(law.locations**2*(1-np.tanh(coordinate)**2))[:,:,None]*np.eye(16)
c1=np.einsum('bk,ki,kj->bij',p,law.means,law.means,optimize=True)-m1[:,:,None]*m1[:,None,:]
delta=m1-m0;between=w[:,:,None]*(1-w[:,:,None])*delta[:,:,None]*delta[:,None,:]
within=(1-w[:,:,None])*c0+w[:,:,None]*c1;cov=within+between
eta=t*(1-t)/d**2
between_response=eta[:,:,None]*between;total_response=eta[:,:,None]*cov
between_energy=np.sum(between_response**2,axis=(1,2));total_energy=np.sum(total_response**2,axis=(1,2))
uncertainty=(w*(1-w))[:,0]
errors={}
for family in study.FAMILIES:
    with np.load(root/'output/runs'/family/'field_0005000.npz') as data:pred=data['prediction']
    errors[family]=np.mean((pred.astype(np.float64)-oracle)**2,axis=1)
bins=(0,.01,.05,.10,.20,.251);rows=[]
for lower,upper in zip(bins[:-1],bins[1:]):
    keep=(uncertainty>=lower)&(uncertainty<upper);count=int(keep.sum())
    if count:
        rows.append(dict(branch_variance_lower=lower,branch_variance_upper=upper,count=count,
            mean_external_time=float(t[keep].mean()),between_jacobian_energy_mean=float(between_energy[keep].mean()),
            **{family+'_mse':float(value[keep].mean()) for family,value in errors.items()}))
with (out/'posterior-bins.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
index=int(np.argmax(between_energy))
study.write(out/'case.json',dict(selection='largest between-branch Jacobian Frobenius norm among all20000 validation anchors',
    index=index,time=float(t[index,0]),posterior_code_weight=float(w[index,0]),x=x[index].tolist(),
    endpoint_branch_mean_difference=delta[index].tolist(),between_jacobian_frobenius=float(np.sqrt(between_energy[index])),
    model_mse={family:float(value[index]) for family,value in errors.items()}))
study.write(out/'summary.json',dict(descriptive_not_registered_selection=True,
    field_count=len(x),between_over_total_covariance_response_energy=float(between_energy.sum()/total_energy.sum()),
    mean_branch_variance=float(uncertainty.mean()),note='Energy ratio is not an additive variance fraction; covariance terms have cross inner products.',
    conditional_pair_label_noise_mse=float(np.mean((np.load(root/'banks/field.npz')['pair'].astype(np.float64)-oracle)**2))))
print(study.load(out/'summary.json'))
