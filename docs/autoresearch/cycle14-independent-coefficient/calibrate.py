"""Pre-outcome architecture/gradient/cost calibration against native initialization."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
parent=args.parent.resolve();doc=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frame_parent',parent/'source/frame_study.py')
prior=importlib.util.module_from_spec(spec);sys.modules['frame_parent']=prior;spec.loader.exec_module(prior)
_,_,native,core,_=prior.setup(parent);prior.verify_inputs(parent)
spec=importlib.util.spec_from_file_location('coefficient_prototype',doc.parents[2]/'synthetic/coefficient_jet.py')
operator=importlib.util.module_from_spec(spec);sys.modules['coefficient_prototype']=operator;spec.loader.exec_module(operator)
from verify_jet import verify
math=verify(operator,operator.OriginalODELayer)
rows=[]
for family,mode in [('jet_c','quadratic'),('affine_c','affine'),('free_c','free'),('direct_b','direct'),('swiglu258',None)]:
    reference=native.old_build(128,'ode_k2','direct').raw
    before=torch.get_rng_state().clone()
    if mode is not None:
        model=reference
        for block in model.blocks:
            for name in ('ode1','ode2'): setattr(block.ffn,name,operator.CoefficientJetLayer(getattr(block.ffn,name),mode))
        assert torch.equal(before,torch.get_rng_state())
    else:
        from synthetic.models import VelocityNet
        from synthetic.recurrent_depth import copy_surrounding
        torch.manual_seed(0);model=VelocityNet(128,'swiglu',width=64,depth=3,ffn_hidden=258)
        copy_surrounding(model,reference)
    model=core.CappedHead(model,'clean');cost=native.compute_budget(model)
    rows.append(dict(family=family,**cost))
prior.verify_inputs(parent)
args.output.write_text(json.dumps(dict(status='PASS',mathematics=math,costs=rows),indent=2)+'\n')
print(json.dumps(dict(mathematics=math,costs=rows),indent=2))
