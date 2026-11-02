"""Matched-state singular-spectrum counterexample; no training."""
import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--save',required=True);a=p.parse_args()
torch.manual_seed(0);matrix=torch.randn(64,64,dtype=torch.float64)
h=torch.linspace(-2,2,64,dtype=torch.float64);h[32]=0
raw=matrix*h[None,:];absolute=matrix*h.abs()[None,:]
sv=float((torch.linalg.svdvals(raw)-torch.linalg.svdvals(absolute)).abs().max())
gram=float((raw@raw.T-absolute@absolute.T).abs().max());assert sv<1e-12 and gram<1e-12
f=lambda x:x+matrix@(x*x);g=lambda x:x+matrix@(x*x.abs())
jf=torch.autograd.functional.jacobian(f,h);jg=torch.autograd.functional.jacobian(g,h);eye=torch.eye(64,dtype=torch.float64)
torch.testing.assert_close(jf-eye,2*raw,atol=1e-12,rtol=1e-12)
torch.testing.assert_close(jg-eye,2*absolute,atol=1e-12,rtol=1e-12)
row=dict(conditional_singular_value_max_error=sv,conditional_gram_max_error=gram,increment_jacobian_singular_value_max_error=float((torch.linalg.svdvals(jf-eye)-torch.linalg.svdvals(jg-eye)).abs().max()),complete_jacobian_norm_raw=float(torch.linalg.matrix_norm(jf,ord=2)),complete_jacobian_norm_abs=float(torch.linalg.matrix_norm(jg,ord=2)),interpretation='Seed0 random64D algebra witness including zero coordinate; no trained model or quality claim.')
out=Path(a.save);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(row,indent=2)+'\n');print(out)
