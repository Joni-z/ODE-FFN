"""Check all eight ODE factorial arms against one nonzero-head initial function."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.models import VelocityNet


def check(device):
    previous=torch.get_default_dtype();rows=[]
    torch.set_num_threads(1)
    try:
        for dtype,tolerance in ((torch.float64,2e-13),(torch.float32,3e-6)):
            torch.set_default_dtype(dtype)
            generator=torch.Generator().manual_seed(250000)
            head=torch.randn((16,64),generator=generator)*.2
            x=torch.randn((128,16),generator=generator).to(device)
            t=torch.rand((128,1),generator=generator).to(device)
            reference=None;reference_rng=None
            for kind,hidden in (('ode_original',64),('ode_vector',32)):
                for pattern in ('map_map','map_increment','increment_map','increment_increment'):
                    torch.manual_seed(0)
                    model=VelocityNet(16,kind,orders=1,gate_hidden=hidden,gate_reference_hidden=64,
                        ode_solve_init='constant_paired',ode_solve_pattern=pattern).to(device)
                    if reference_rng is None:reference_rng=torch.get_rng_state()
                    else:torch.testing.assert_close(torch.get_rng_state(),reference_rng,atol=0,rtol=0)
                    with torch.no_grad():model.output.weight.copy_(head.to(device))
                    value=model(x,t)
                    if reference is None:reference=value.detach().clone()
                    error=float((value-reference).abs().max().detach())
                    torch.testing.assert_close(value,reference,atol=tolerance,rtol=tolerance)
                    value.square().mean().backward()
                    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
                    rows.append(dict(model=kind,pattern=pattern,dtype=str(dtype),maximum_forward_difference=error))
    finally:
        torch.set_default_dtype(previous)
    return dict(state='PASS',device=device,rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    print(json.dumps(check(parser.parse_args().device)))
