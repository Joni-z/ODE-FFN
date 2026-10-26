"""Measure whole-network paired initialization defects with a nonzero head."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.models import VelocityNet


def check(device):
    previous=torch.get_default_dtype()
    rows=[]
    torch.set_num_threads(1)
    try:
        for dtype,tolerance in ((torch.float64,2e-13),(torch.float32,3e-6)):
            torch.set_default_dtype(dtype)
            for kind,hidden in (('ode_original',64),('ode_vector',32)):
                options=dict(orders=1,gate_hidden=hidden,gate_reference_hidden=64,
                             ode_solve_init='constant_paired')
                torch.manual_seed(0);parent=VelocityNet(16,kind,**options).to(device)
                torch.manual_seed(0);increment=VelocityNet(16,kind,**options,
                    ode_solve_return='increment').to(device)
                with torch.no_grad():
                    head=torch.randn_like(parent.output.weight)*.2
                    parent.output.weight.copy_(head);increment.output.weight.copy_(head)
                x=torch.randn(128,16,device=device);t=torch.rand(128,1,device=device)
                a,b=parent(x,t),increment(x,t)
                error=float((a-b).abs().max().detach())
                torch.testing.assert_close(a,b,atol=tolerance,rtol=tolerance)
                for value in (a,b):assert torch.isfinite(value).all()
                (b.square().mean()).backward()
                assert all(torch.isfinite(p.grad).all() for p in increment.parameters())
                rows.append(dict(model=kind,dtype=str(dtype),maximum_forward_difference=error,
                                 atol=tolerance,rtol=tolerance))
    finally:
        torch.set_default_dtype(previous)
    return dict(state='PASS',device=device,rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    print(json.dumps(check(parser.parse_args().device)))
