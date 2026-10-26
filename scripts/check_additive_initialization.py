"""Check the registered full-width additive/multiplicative initialization."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_architecture_study import macs
from synthetic.models import VelocityNet


def check():
    torch.set_num_threads(1)
    previous=torch.get_default_dtype();rows=[]
    try:
        for dtype in (torch.float64,torch.float32):
            torch.set_default_dtype(dtype)
            for dim in (2,16):
                gen=torch.Generator().manual_seed(260000)
                head=torch.randn(dim,64,generator=gen)*.2
                x=torch.randn(128,dim,generator=gen)
                t=torch.rand(128,1,generator=gen)
                for kind,hidden in (('ode_original',64),('ode_vector',32)):
                    pair=[];states=[]
                    for mode in ('multiplicative','additive'):
                        torch.manual_seed(0)
                        model=VelocityNet(dim,kind,orders=1,gate_hidden=hidden,
                            gate_reference_hidden=64,ode_solve_init='constant_paired',ode_operator_mode=mode)
                        states.append(torch.get_rng_state());pair.append(model)
                        with torch.no_grad():model.output.weight.copy_(head)
                    a,b=pair
                    torch.testing.assert_close(states[0],states[1],atol=0,rtol=0)
                    for name,value in a.state_dict().items():
                        torch.testing.assert_close(value,b.state_dict()[name],atol=0,rtol=0)
                    y,z=a(x,t),b(x,t)
                    torch.testing.assert_close(y,z,atol=0,rtol=0)
                    for model in pair:
                        model(x,t).square().mean().backward()
                        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                    other=dict(b.named_parameters());gradient_error=0.
                    for name,value in a.named_parameters():
                        if '.t_func.2.' not in name:
                            gradient_error=max(gradient_error,float((value.grad-other[name].grad).abs().max()))
                            tol=2e-12 if dtype==torch.float64 else 3e-6
                            torch.testing.assert_close(value.grad,other[name].grad,atol=tol,rtol=tol)
                    assert a.parameter_count()==b.parameter_count() and macs(a,dim)==macs(b,dim)
                    rows.append(dict(dim=dim,kind=kind,dtype=str(dtype),forward_difference=0.,
                        inherited_gradient_maximum_difference=gradient_error,parameters=a.parameter_count(),
                        forward_macs=macs(a,dim)))
    finally:torch.set_default_dtype(previous)
    return dict(state='PASS',device='cpu',torch=torch.__version__,rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=check()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
