"""Finite-difference, actual-trainer and old-gradient compatibility checks."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.conservative_readout import build as parent_build,ReadoutNet,compute_budget
from synthetic.directional_readout import build,DirectionalReadout
from synthetic.train import restore_model
from scripts.run_derivative_orientation import ARMS,train_cell,harness,common
from scripts.check_gate_geometry import jacobian


def same(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


def run(output,data,parent):
    torch.set_num_threads(1)
    output.mkdir(parents=True,exist_ok=False)
    hashes=json.loads((ROOT/'source-sha256.json').read_text())
    for name,value in hashes.items():assert common.sha256(ROOT/name)==value
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests',
        '-p','test_directional_readout.py','-v'],cwd=ROOT,capture_output=True,text=True,timeout=120)
    (output/'unit-tests.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('directional unit tests failed')
    dataset='rotated_product_bimodal_16d'
    config=dict(seed=0,steps=3,datasets=[dataset],data_roots={dataset:str(data)})
    config_path=output/'study.json';common.write(config_path,config)
    common.prepare_banks(config,dataset,output)
    # Actual default trainer path after dependency injection refactor.
    for arm in ('ode_k2_gradient','swiglu192_gradient'):
        harness.train_cell(config_path,output,dataset,arm,steps=3)
        new=torch.load(output/'runs'/dataset/arm/'step_0000003.pt',map_location='cpu',weights_only=False)
        old=torch.load(parent/'preflight/runs'/dataset/arm/'step_0000003.pt',map_location='cpu',weights_only=False)
        for key in ('model','optimizer','runtime_buffers','data_rng','torch_rng','input_stream_digest'):same(new[key],old[key])
    rows=[]
    x=torch.linspace(-1.,1.,512*16).reshape(512,16);t=torch.full((512,1),.4)
    for arm,setting in ARMS.items():
        model=build(16,setting['family'],'directional')
        costs=compute_budget(model)
        with torch.no_grad():
            for _ in range(3):model(x,t)
            start=time.perf_counter()
            for _ in range(10):model(x,t)
        seconds=(time.perf_counter()-start)/10
        train_cell(config_path,output,dataset,arm,3)
        folder=output/'runs'/dataset/arm
        result=json.loads((folder/'result.json').read_text())
        state=torch.load(folder/'step_0000003.pt',map_location='cpu',weights_only=False)
        model=build(16,setting['family'],'directional');restore_model(model,state)
        replay=common.field_evaluation(model,output/'banks'/dataset)
        for k,v in replay.items():assert v==result[k]
        initial=torch.load(folder/'step_0000000.pt',map_location='cpu',weights_only=False)
        reference=torch.load(parent/'output/runs'/dataset/(setting['family']+'_gradient')/'step_0000000.pt',map_location='cpu',weights_only=False)
        for k in ('model','runtime_buffers','data_rng','torch_rng'):same(initial[k],reference[k])
        rows.append(dict(arm=arm,costs=costs,isolated_batch512_seconds=seconds,
            stream_digest=result['input_stream_digest'],reload='exact'))
        print(json.dumps(rows[-1]),flush=True)
    assert len({r['stream_digest'] for r in rows})==1
    # Address the new concern on all old gradient backbones/tasks/times.
    fd=[];maximum_value_difference=maximum_jacobian_difference=0.
    for task in common.DATASETS:
        banks=parent/'output/banks'/task
        for family in ('ode_k2','swiglu128','swiglu192'):
            checkpoint=parent/'output/runs'/task/(family+'_gradient')/'step_0005000.pt'
            with np.load(banks/'panel_0.5.npz') as data:dim=data['x'].shape[1]
            model=parent_build(dim,family,'gradient').double().eval()
            restore_model(model,torch.load(checkpoint,map_location='cpu',weights_only=False))
            for parameter in model.parameters():parameter.requires_grad_(False)
            for clock in (.1,.3,.5,.7,.9):
                with np.load(banks/f'panel_{clock:.1f}.npz') as data:
                    xx,tt=[torch.from_numpy(data[k][:1]).double() for k in ('x','t')]
                direction=torch.linspace(-1.,1.,dim,dtype=torch.float64).reshape(1,-1)
                direction=direction/direction.norm()
                _,jac=jacobian(model,xx,tt)
                h=1e-5
                with torch.no_grad():finite=(model(xx+h*direction,tt)-model(xx-h*direction,tt))/(2*h)
                analytic=torch.einsum('bij,bj->bi',jac,direction)
                error=float((finite-analytic).norm()/max(1.,float(analytic.norm())))
                assert error<1e-5,(task,family,clock,error)
                fd.append(dict(dataset=task,family=family,time=clock,relative_directional_error=error))
            # Explicit normalization has the native primal and same calculus.
            if task==dataset:
                with np.load(banks/'panel_0.5.npz') as data:
                    xx,tt=[torch.from_numpy(data[k][:4]).double() for k in ('x','t')]
                modified=DirectionalReadout(copy.deepcopy(model.raw),dim)
                repaired_A=ReadoutNet(modified.raw,dim,'gradient')
                av,aj=jacobian(model,xx,tt);bv,bj=jacobian(repaired_A,xx,tt)
                torch.testing.assert_close(av,bv,rtol=1e-9,atol=1e-10)
                torch.testing.assert_close(aj,bj,rtol=1e-8,atol=1e-9)
                maximum_value_difference=max(maximum_value_difference,float((av-bv).abs().max()))
                maximum_jacobian_difference=max(maximum_jacobian_difference,float((aj-bj).abs().max()))
    common.write(output/'parent-finite-differences.json',fd)
    for name,value in hashes.items():assert common.sha256(ROOT/name)==value
    common.write(output/'PASS.json',dict(status='PASS',unit_methods=7,directional_trainers=rows,
        default_parent_histories='two actual step3 model/optimizer/RNG states exact',
        parent_potential_finite_differences=len(fd),maximum_parent_fd_error=max(r['relative_directional_error'] for r in fd),
        normalization_potential_value_difference=maximum_value_difference,
        normalization_potential_jacobian_difference=maximum_jacobian_difference,source_sha256=hashes))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    a=p.parse_args();run(a.output.resolve(),a.data.resolve(),a.parent.resolve())
