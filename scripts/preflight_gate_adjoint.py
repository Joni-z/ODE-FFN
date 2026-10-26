"""Validate actual gate-removal trainers and repeated CLI execution."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.gate_adjoint import build
from synthetic.train import restore_model
from scripts.run_gate_adjoint import ARMS,train_cell,harness,common


def same(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b)
        for aa,bb in zip(a,b):same(aa,bb)
    else:assert a==b


def main(output,data,parent):
    torch.set_num_threads(1)
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((ROOT/'source-sha256.json').read_text())
    for name,value in manifest.items():assert common.sha256(ROOT/name)==value
    unit=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_*adjoint.py','-v'],
        cwd=ROOT,capture_output=True,text=True,timeout=120)
    (output/'unit-tests.log').write_text(unit.stdout+unit.stderr)
    if unit.returncode:raise RuntimeError('adjoint unit tests failed')
    dataset=common.DATASETS[2]
    config=dict(seed=0,steps=3,datasets=[dataset],data_roots={dataset:str(data)})
    path=output/'study.json';common.write(path,config)
    common.prepare_banks(config,dataset,output)
    for bank in (output/'banks'/dataset).glob('*.npz'):
        with np.load(bank) as a,np.load(parent/'output/banks'/dataset/bank.name) as b:
            assert a.files==b.files
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    records=[]
    for arm,setting in ARMS.items():
        train_cell(path,output,dataset,arm,3)
        folder=output/'runs'/dataset/arm
        initial=torch.load(folder/'step_0000000.pt',map_location='cpu',weights_only=False)
        old=torch.load(parent/'output/runs'/dataset/(setting['family']+'_gradient')/'step_0000000.pt',map_location='cpu',weights_only=False)
        for key in ('model','runtime_buffers','data_rng','torch_rng'):same(initial[key],old[key])
        final=torch.load(folder/'step_0000003.pt',map_location='cpu',weights_only=False)
        model=build(16,setting['family'],setting['readout']);restore_model(model,final)
        actual=common.field_evaluation(model,output/'banks'/dataset)
        result=json.loads((folder/'result.json').read_text())
        for key,value in actual.items():assert value==result[key]
        for step in (0,3):
            visits=json.loads((folder/f'visits_{step:07d}.json').read_text())
            assert len(visits)==(6 if setting['family']=='ode_k2' else 0)
        records.append(dict(arm=arm,reload='exact',stream_digest=result['input_stream_digest']))
    assert len({r['stream_digest'] for r in records})==1
    # Default trainer behavior is unchanged by the optional visit collector.
    default=output/'default-compatibility';default.mkdir()
    (default/'banks').symlink_to(output/'banks',target_is_directory=True)
    for arm in ('ode_k2_gradient','swiglu192_gradient'):
        harness.train_cell(path,default,dataset,arm,3)
        aa=torch.load(default/'runs'/dataset/arm/'step_0000003.pt',map_location='cpu',weights_only=False)
        bb=torch.load(parent/'preflight/runs'/dataset/arm/'step_0000003.pt',map_location='cpu',weights_only=False)
        for key in ('model','optimizer','runtime_buffers','data_rng','torch_rng','input_stream_digest'):same(aa[key],bb[key])
    smoke=output/'cli';smoke.mkdir();(smoke/'banks').symlink_to(output/'banks',target_is_directory=True)
    for arm in ('ode_k2_gradient','swiglu192_gate_omitted'):
        with (smoke/(arm+'.log')).open('w') as log:
            child=subprocess.run([sys.executable,str(ROOT/'scripts/run_gate_adjoint.py'),'cell','--output',str(smoke),
                '--config',str(path),'--dataset',dataset,'--arm',arm],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        if child.returncode:raise RuntimeError('CLI smoke failed: '+arm)
        aa=torch.load(smoke/'runs'/dataset/arm/'step_0000003.pt',map_location='cpu',weights_only=False)
        bb=torch.load(output/'runs'/dataset/arm/'step_0000003.pt',map_location='cpu',weights_only=False)
        for key in ('model','optimizer','runtime_buffers','data_rng','torch_rng','input_stream_digest'):same(aa[key],bb[key])
    for name,value in manifest.items():assert common.sha256(ROOT/name)==value
    common.write(output/'PASS.json',dict(status='PASS',unit_methods=12,actual_trainers=records,
        default_parent_histories='two exact step3 histories',cli_repeats='two exact model/optimizer/RNG histories',
        source_manifest_sha256=common.sha256(ROOT/'source-sha256.json')))
    print(json.dumps(dict(status='PASS',actual_trainers=len(records),unit_methods=12)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    a=p.parse_args();main(a.output.resolve(),a.data.resolve(),a.parent.resolve())
