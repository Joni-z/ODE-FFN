"""Ideal fixed-D local metric witness; not a native optimizer experiment."""
import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--save',required=True);a=p.parse_args()
torch.manual_seed(0);gradient=torch.randn(32,32,dtype=torch.double)
d=torch.linspace(.5,2,32,dtype=torch.double);c=torch.linspace(.25,4,32,dtype=torch.double)
def polar(matrix):
    u,_,v=torch.linalg.svd(matrix,full_matrices=False)
    return u@v
effective=-polar(gradient/d[None,:]);raw=-polar(gradient)*d[None,:]
prime=-polar((gradient*c[None,:])/(c*d)[None,:]);error=float((effective-prime).abs().max());assert error<1e-12
objective=float((gradient/d[None,:]*effective).sum());optimum=-float(torch.linalg.svdvals(gradient/d[None,:]).sum());assert abs(objective-optimum)<1e-10
row=dict(effective_step_coordinate_max_error=error,linear_objective=objective,negative_nuclear_norm=optimum,ideal_effective_step_sigma_max=float(torch.linalg.matrix_norm(effective,ord=2)),raw_coordinate_muon_effective_sigma_max=float(torch.linalg.matrix_norm(raw,ord=2)),scope='Seed0 32D fixed invertible-D exact-SVD matrix identity; not native Muon or training.')
out=Path(a.save);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(row,indent=2)+'\n');print(out)
