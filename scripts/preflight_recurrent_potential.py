"""Actual trainer checks for the shared/cloned complete-potential factor."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.recurrent_potential import build
from synthetic.conservative_readout import build as old_build
from synthetic.train import restore_model
from scripts.run_recurrent_potential import ARMS,train_cell,common


def same(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:same(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b)
        for aa,bb in zip(a,b):same(aa,bb)
    else:assert a==b


def partial_initial_names(new,old):
    for name,value in new.items():
        assert name.startswith('raw.')
        mapped=name[4:]
        for index in (1,2):mapped=mapped.replace(f'.blocks.{index}.','.blocks.0.')
        assert torch.equal(value,old[mapped]),name


def main(output,data,parent,recurrent):
    torch.set_num_threads(1)
    output.mkdir(parents=True,exist_ok=False)
    hashes=json.loads((ROOT/'source-sha256.json').read_text())
    for name,value in hashes.items():assert common.sha256(ROOT/name)==value
    unit=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_recurrent_potential.py','-v'],
        cwd=ROOT,capture_output=True,text=True,timeout=120)
    (output/'unit-tests.log').write_text(unit.stdout+unit.stderr)
    if unit.returncode:raise RuntimeError('unit tests failed')
    dataset=common.DATASETS[2]
    config=dict(seed=0,steps=3,datasets=[dataset],data_roots={dataset:str(data)})
    path=output/'study.json';common.write(path,config);common.prepare_banks(config,dataset,output)
    for bank in (output/'banks'/dataset).glob('*.npz'):
        for reference in (parent/'output/banks'/dataset/bank.name,recurrent/'output-product/banks'/dataset/bank.name):
            with np.load(bank) as a,np.load(reference) as b:
                assert a.files==b.files
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    # New native wrapper executes exactly the old three independent blocks.
    for family in ('ode_k2','swiglu128','swiglu192'):
        checkpoint=parent/'output/runs'/dataset/(family+'_gradient')/'step_0005000.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        old=old_build(16,family,'gradient').double();restore_model(old,state)
        new=build(16,family,'native_gradient').double();restore_model(new.raw,state)
        with np.load(output/'banks'/dataset/'panel_0.5.npz') as bank:
            x,t=[torch.from_numpy(bank[k][:3]).double() for k in ('x','t')]
        xx=x.requires_grad_(True)
        a,b=old(xx,t),new(xx,t)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        ja=torch.stack([torch.autograd.grad(a[:,j].sum(),xx,retain_graph=True)[0] for j in range(16)],1)
        jb=torch.stack([torch.autograd.grad(b[:,j].sum(),xx,retain_graph=True)[0] for j in range(16)],1)
        torch.testing.assert_close(ja,jb,rtol=0,atol=0)
    records=[]
    for arm,setting in ARMS.items():
        train_cell(path,output,dataset,arm,3)
        folder=output/'runs'/dataset/arm
        zero=torch.load(folder/'step_0000000.pt',map_location='cpu',weights_only=False)
        original=torch.load(parent/'output/runs'/dataset/(setting['family']+'_gradient')/'step_0000000.pt',map_location='cpu',weights_only=False)
        partial_initial_names(zero['model'],original['model'])
        partial_initial_names(zero['runtime_buffers'],original['runtime_buffers'])
        for key in ('data_rng','torch_rng'):same(zero[key],original[key])
        final=torch.load(folder/'step_0000003.pt',map_location='cpu',weights_only=False)
        model=build(16,setting['family'],setting['readout']);restore_model(model,final)
        result=json.loads((folder/'result.json').read_text())
        replay=common.field_evaluation(model,output/'banks'/dataset)
        for key,value in replay.items():assert value==result[key]
        visits=json.loads((folder/'visits_0000003.json').read_text())
        assert len(visits)==(6 if setting['family']=='ode_k2' else 0)
        records.append(dict(arm=arm,reload='exact',stream_digest=result['input_stream_digest']))
    assert len({r['stream_digest'] for r in records})==1
    smoke=output/'cli';smoke.mkdir();(smoke/'banks').symlink_to(output/'banks',target_is_directory=True)
    for arm in ('ode_k2_shared_gradient','swiglu128_cloned_direct'):
        with (smoke/(arm+'.log')).open('w') as log:
            code=subprocess.run([sys.executable,str(ROOT/'scripts/run_recurrent_potential.py'),'cell',
                '--output',str(smoke),'--config',str(path),'--dataset',dataset,'--arm',arm],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
        if code:raise RuntimeError('CLI child failed: '+arm)
        a=torch.load(smoke/'runs'/dataset/arm/'step_0000003.pt',map_location='cpu',weights_only=False)
        b=torch.load(output/'runs'/dataset/arm/'step_0000003.pt',map_location='cpu',weights_only=False)
        for key in ('model','optimizer','runtime_buffers','data_rng','torch_rng','input_stream_digest'):same(a[key],b[key])
    for name,value in hashes.items():assert common.sha256(ROOT/name)==value
    common.write(output/'PASS.json',dict(status='PASS',unit_methods=4,actual_trainers=records,
        native_potential_compatibility='three actual learned value/Jacobian probes bitwise',
        cli_repeats='two complete model/optimizer/RNG histories exact'))
    print(json.dumps(dict(status='PASS',actual_trainers=len(records))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--recurrent',type=Path,required=True)
    a=p.parse_args();main(a.output.resolve(),a.data.resolve(),a.parent.resolve(),a.recurrent.resolve())
