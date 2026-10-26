"""Verify that the default model retains the frozen predecessor's trajectory."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.models import VelocityNet


def check(reference_file):
    spec=importlib.util.spec_from_file_location('frozen_rate_order_reference',reference_file)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    torch.set_num_threads(1)
    configurations=[('ode_original',dict(orders=1,gate_hidden=64)),
                         ('ode_vector',dict(orders=1,gate_hidden=32,gate_reference_hidden=64)),
                         ('swiglu',dict(ffn_hidden=96)),('swiglu',dict(ffn_hidden=128))]
    for kind,hidden in (('ode_original',64),('ode_vector',32)):
        for mode in ('map','increment'):
            configurations.append((kind,dict(orders=1,gate_hidden=hidden,ode_solve_init='constant_paired',
                                            ode_solve_return=mode)))
    for kind,options in configurations:
        torch.manual_seed(0);reference=old.VelocityNet(16,kind,**options)
        previous_rng=torch.get_rng_state()
        torch.manual_seed(0);current=VelocityNet(16,kind,**options)
        torch.testing.assert_close(previous_rng,torch.get_rng_state(),atol=0,rtol=0)
        generator=torch.Generator().manual_seed(100000)
        a=torch.optim.AdamW(reference.parameters(),lr=3e-4,weight_decay=0.)
        b=torch.optim.AdamW(current.parameters(),lr=3e-4,weight_decay=0.)
        # Nonzero head exercises the whole backbone from the first update.
        with torch.no_grad():
            head=torch.randn(current.output.weight.shape,generator=generator)
            reference.output.weight.copy_(head);current.output.weight.copy_(head)
        for _ in range(3):
            x=torch.randn(31,16,generator=generator)
            t=torch.rand(31,1,generator=generator)
            target=torch.randn(31,16,generator=generator)
            torch.testing.assert_close(reference(x,t),current(x,t),atol=0,rtol=0)
            for model,optimizer in ((reference,a),(current,b)):
                optimizer.zero_grad();(model(x,t)-target).square().mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            for name,value in reference.state_dict().items():
                torch.testing.assert_close(value,current.state_dict()[name],atol=0,rtol=0)
            for pa,pb in zip(reference.parameters(),current.parameters()):
                for key in a.state[pa]:
                    torch.testing.assert_close(a.state[pa][key],b.state[pb][key],atol=0,rtol=0)
    return dict(state='PASS',models=len(configurations),updates_per_model=3,bit_identical=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-file',type=Path,required=True)
    print(json.dumps(check(parser.parse_args().reference_file)))
