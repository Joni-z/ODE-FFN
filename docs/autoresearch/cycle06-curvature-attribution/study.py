"""Fixed read-only full-network curvature counterfactuals and tangent checks."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.func import functional_call


def load(path: Path) -> dict:return json.loads(path.read_text())
def sha(path: Path) -> str:return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path: Path,value: object) -> None:path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify_bindings(root: Path) -> None:
    for name,digest in load(root/'bindings.json').items():assert sha(Path(name))==digest,name
    base=Path(load(root/'settings.json')['cycle05'])
    for name,digest in load(base/'sealed-runtime.json').items():assert sha(base/name)==digest,name


def prepare(root: Path,base: Path) -> None:
    root.mkdir(exist_ok=False);source=root/'source';source.mkdir()
    for name in ('protocol.md','study.py'):shutil.copyfile(Path(__file__).parent/name,source/name)
    repo=Path(__file__).resolve().parents[3]
    write(root/'settings.json',dict(cycle05=str(base.resolve()),registration_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()))
    paths=list(source.iterdir())+[root/'settings.json',base/'sealed-runtime.json']
    write(root/'bindings.json',{str(p):sha(p) for p in paths})
    print(dict(prepared=str(root)),flush=True)


def table(path: Path,rows: list[dict]) -> None:
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def tangent_checks(model,x: torch.Tensor,t: torch.Tensor,law: str) -> list[dict]:
    model=copy.deepcopy(model).double()
    with torch.no_grad():
        for block in model.raw.blocks:block.ffn.transport.zero_()
    named=dict(model.named_parameters());names=list(named);params=tuple(p.detach() for p in named.values());rows=[]
    def function(*values):return functional_call(model,dict(zip(names,values)),(x.double(),t.double()))
    for selected in ((0,),(1,),(2,),(0,1,2)):
        c_tangent={name:torch.zeros_like(p) for name,p in named.items()}
        ordinary={name:torch.zeros_like(p) for name,p in named.items()}
        for index in selected:
            module=model.raw.blocks[index].ffn;prefix=f'raw.blocks.{index}.ffn.';m=module.left_width;r=module.w3.in_features
            delta=.01*torch.sin(torch.arange(module.transport.numel(),dtype=torch.float64)+index).reshape_as(module.transport)
            c_tangent[prefix+'transport']=delta
            ordinary[prefix+'w12.weight'][:m]=.5*delta@module.w12.weight[m:r].detach()
            ordinary[prefix+'w12.bias'][:m]=.5*delta@module.w12.bias[m:r].detach()
        _,left=torch.autograd.functional.jvp(function,params,tuple(c_tangent.values()))
        _,right=torch.autograd.functional.jvp(function,params,tuple(ordinary.values()))
        torch.testing.assert_close(left,right,rtol=1e-9,atol=1e-10)
        assert float(left.abs().max())>1e-10,(law,selected,'vacuous tangent')
        rows.append(dict(law=law,blocks='+'.join(map(str,selected)),anchors=len(x),
            C_tangent_rms=float(left.square().mean().sqrt()),ordinary_tangent_rms=float(right.square().mean().sqrt()),
            max_absolute_difference=float((left-right).abs().max())))
    return rows


def run(root: Path) -> None:
    started=time.monotonic();verify_bindings(root);settings=load(root/'settings.json');base=Path(settings['cycle05'])
    sys.path.insert(0,str(base/'source'));import study as previous
    config,native,quadrature,midpoint=previous.setup(base);previous.verify_inputs(base,native);torch.set_num_threads(1)
    out=root/'output';out.mkdir(exist_ok=False);fields=[];panels=[];cases=[];tangents=[]
    write(out/'status.json',dict(state='RUNNING',pid=os.getpid()))
    for entry in config['laws']:
        law=entry['law'];banks=Path(entry['banks']);folder=base/'output/runs'/law/'quadrature2'
        model=previous.build(base,law,'quadrature2');native.restore_model(model,torch.load(folder/'step_0005000.pt',weights_only=False));model.eval()
        with np.load(banks/'probe.npz') as data:tangents+=tangent_checks(model,torch.from_numpy(data['x'][:32]),torch.from_numpy(data['t'][:32]),law)
        counterfactual=copy.deepcopy(model)
        for block in counterfactual.raw.blocks:
            old=block.ffn;new=midpoint(old.w12,old.w3)
            with torch.no_grad():new.transport.copy_(old.transport)
            block.ffn=new
        for name,value in model.state_dict().items():assert torch.equal(value,counterfactual.state_dict()[name]),name
        actual=native.legacy.field_evaluation(model,banks,save=out/f'{law}-actual.npz')
        changed=native.legacy.field_evaluation(counterfactual,banks,save=out/f'{law}-midpoint.npz')
        with np.load(folder/'field_0005000.npz') as data, np.load(out/f'{law}-actual.npz') as now:np.testing.assert_array_equal(data['prediction'],now['prediction'])
        with np.load(out/f'{law}-actual.npz') as data:fq=data['prediction'].astype(np.float64)
        with np.load(out/f'{law}-midpoint.npz') as data:fm=data['prediction'].astype(np.float64)
        with np.load(banks/'field.npz') as data:truth=data['oracle'].astype(np.float64);clock=data['t'];inputs=data['x']
        assert len(fq)==(10000 if entry['dim']==2 else 20000)
        error=fm-truth;delta=fq-fm;alignment=2*float(np.mean(error*delta));energy=float(np.mean(delta**2))
        risk_q=float(np.mean((fq-truth)**2));risk_m=float(np.mean(error**2));effect=risk_q-risk_m
        assert np.isclose(effect,alignment+energy,rtol=1e-9,atol=1e-12)
        np.testing.assert_allclose([risk_q,risk_m],[actual['oracle_mse'],changed['oracle_mse']],rtol=2e-6,atol=1e-9)
        fields.append(dict(law=law,rows=len(fq),actual_MSE=risk_q,midpoint_counterfactual_MSE=risk_m,
            actual_minus_counterfactual_MSE=effect,alignment_cross_term=alignment,displacement_energy=energy,
            actual_vs_counterfactual_relative_prediction_rms=float(np.sqrt(np.mean(delta**2)/np.mean(fq**2))),
            decomposition_absolute_error=abs(effect-alignment-energy)))
        local=np.mean((fq-truth)**2,axis=1)-np.mean(error**2,axis=1)
        indices=np.lexsort((np.arange(len(local)),-np.abs(local)))[:3]
        for rank,index in enumerate(indices):cases.append(dict(law=law,rank_by_absolute_error_change=rank+1,row=int(index),time=float(clock[index,0]),
            input_norm=float(np.linalg.norm(inputs[index])),actual_MSE=float(np.mean((fq[index]-truth[index])**2)),
            midpoint_counterfactual_MSE=float(np.mean(error[index]**2)),actual_minus_counterfactual_MSE=float(local[index]),
            displacement_rms=float(np.sqrt(np.mean(delta[index]**2)))))
        for clock in previous.TIMES:
            with np.load(banks/f'panel_{clock:.1f}.npz') as data:x,t,truth,jtrue=[data[k] for k in ('x','t','oracle','oracle_jacobian')]
            aq,aj=native.legacy.jacobian(model,torch.from_numpy(x),torch.from_numpy(t))
            cm,cj=native.legacy.jacobian(counterfactual,torch.from_numpy(x),torch.from_numpy(t))
            with np.load(folder/'diagnostics'/f'panel_{clock:.1f}.npz') as data:
                np.testing.assert_array_equal(aq.numpy(),data['prediction']);np.testing.assert_array_equal(aj.numpy(),data['jacobian'])
            panels.append(dict(law=law,time=clock,anchors=len(x),actual_J_error=float(np.mean(np.sum((aj.numpy()-jtrue)**2,axis=(1,2)))),
                midpoint_counterfactual_J_error=float(np.mean(np.sum((cj.numpy()-jtrue)**2,axis=(1,2)))),
                J_difference_relative_rms=float(((aj.double()-cj.double()).square().sum()/aj.double().square().sum()).sqrt())))
            np.savez_compressed(out/f'{law}-midpoint-panel-{clock:.1f}.npz',prediction=cm.numpy(),jacobian=cj.numpy())
        assert time.monotonic()-started<600,'registered wall-time ceiling'
        write(out/'status.json',dict(state='RUNNING',pid=os.getpid(),completed=len(fields),total=5,elapsed_seconds=time.monotonic()-started))
        print(fields[-1],flush=True)
    assert len(fields)==5 and len(panels)==25 and len(cases)==15 and len(tangents)==20
    verify_bindings(root);previous.verify_inputs(base,native)
    for name,rows in [('field-attribution.csv',fields),('time-panels.csv',panels),('cases.csv',cases),('tangent-identities.csv',tangents)]:table(out/name,rows)
    import psutil
    write(out/'receipt.json',dict(status='PASS',pid=os.getpid(),elapsed_seconds=time.monotonic()-started,
        process_rss_gib=psutil.Process().memory_info().rss/2**30,field_banks=5,actual_field_predictions_replayed=80000,
        actual_jacobian_panels_replayed=25,counterfactual_jacobian_panels=25,complete_tangent_checks=20,
        registration_commit=settings['registration_commit'],no_training_or_nomination=True))
    write(out/'status.json',dict(state='COMPLETE',pid=os.getpid(),completed=5,total=5,elapsed_seconds=time.monotonic()-started))
    print(dict(status='PASS',elapsed=time.monotonic()-started),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--base',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.root,args.base)
    else:run(args.root)
