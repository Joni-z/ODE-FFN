"""Verify and measure the unchanged complete field with an explicit adjoint."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.conservative_readout import build,compute_budget
from synthetic.explicit_adjoint import ExplicitAdjointNet
from synthetic.train import restore_model

DATASETS=('gaussian_aniso_2d','gmm_ring8_2d','rotated_product_bimodal_16d')
FAMILIES=('ode_k2','swiglu128','swiglu192')
TIMES=(.1,.3,.5,.7,.9)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def table(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def field(model,x,t,dtype):
    pieces=[]
    with torch.no_grad():
        for start in range(0,len(x),512):
            pieces.append(model(torch.from_numpy(x[start:start+512]).to(dtype),
                                torch.from_numpy(t[start:start+512]).to(dtype)).numpy())
    return np.concatenate(pieces)


def derivative(model,x,t):
    xx=x.detach().clone().requires_grad_(True)
    value=model(xx,t)
    jac=torch.stack([torch.autograd.grad(value[:,j].sum(),xx,
        retain_graph=j+1<value.shape[1])[0] for j in range(value.shape[1])],1)
    return value.detach().numpy(),jac.detach().numpy()


def chunked_derivative(model,x,t):
    values=[];jacobians=[]
    for start in range(0,len(x),64):
        v,j=derivative(model,torch.from_numpy(x[start:start+64]).double(),
                       torch.from_numpy(t[start:start+64]).double())
        values.append(v);jacobians.append(j)
    return np.concatenate(values),np.concatenate(jacobians)


def make_pair(state,dim,family,dtype=torch.float64):
    reference=build(dim,family,'gradient').to(dtype=dtype).eval()
    restore_model(reference,state)
    explicit=ExplicitAdjointNet(copy.deepcopy(reference.raw)).eval()
    return reference,explicit


def gradients_and_update(reference,explicit,x,t,target):
    xx=torch.from_numpy(x[:3]).double().requires_grad_(True)
    tt=torch.from_numpy(t[:3]).double().requires_grad_(True)
    yy=torch.from_numpy(target[:3]).double()
    prediction_a=reference(xx,tt);prediction_b=explicit(xx,tt)
    torch.testing.assert_close(prediction_a,prediction_b,rtol=1e-10,atol=1e-11)
    ga=torch.autograd.grad((prediction_a-yy).square().mean(),(xx,tt,*reference.parameters()))
    gb=torch.autograd.grad((prediction_b-yy).square().mean(),(xx,tt,*explicit.parameters()))
    maximum=0.
    for a,b in zip(ga,gb):
        torch.testing.assert_close(a,b,rtol=1e-7,atol=1e-9)
        maximum=max(maximum,float((a-b).abs().max()))
    # Optimizer probes use clones; no saved checkpoint or evaluated model changes.
    aa=copy.deepcopy(reference);bb=ExplicitAdjointNet(copy.deepcopy(explicit.raw))
    for model in (aa,bb):
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.)
        optimizer.zero_grad()
        loss=(model(xx.detach(),tt.detach())-yy).square().mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    update_error=0.
    for (name,a),(other,b) in zip(aa.named_parameters(),bb.named_parameters()):
        assert name==other
        torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-10)
        update_error=max(update_error,float((a-b).abs().max().detach()))
    return maximum,update_error


def benchmark(state,dim,family,x,t,target):
    reference,explicit=make_pair(state,dim,family,torch.float32)
    budget_a,budget_b=compute_budget(reference),compute_budget(explicit)
    xx=torch.from_numpy(x[:512]);tt=torch.from_numpy(t[:512]);yy=torch.from_numpy(target[:512])
    output=[]
    for mode in ('inference','loss_and_backward'):
        def operation(model):
            if mode=='inference':
                with torch.no_grad():model(xx,tt)
            else:
                model.zero_grad(set_to_none=True)
                (model(xx,tt)-yy).square().mean().backward()
        for _ in range(5):operation(reference);operation(explicit)
        elapsed={'autograd':[],'explicit':[]}
        for repeat in range(20):
            order=(('autograd',reference),('explicit',explicit))
            if repeat%2:order=tuple(reversed(order))
            for name,model in order:
                start=time.perf_counter();operation(model)
                elapsed[name].append(1000*(time.perf_counter()-start))
        for name in elapsed:
            values=np.array(elapsed[name])
            output.append(dict(method=name,mode=mode,median_ms=float(np.median(values)),
                p25_ms=float(np.quantile(values,.25)),p75_ms=float(np.quantile(values,.75)),
                repetitions=len(values),raw_ms=elapsed[name]))
    return budget_a,budget_b,output


def main(parent,orientation,orientation_results,output):
    torch.set_num_threads(1)
    assert load(orientation/'output/status.json')['state']=='COMPLETE'
    assert load(orientation_results/'audit.json')['status']=='PASS'
    # Source of this evaluator and every local synthetic module is frozen first.
    source_manifest=load(ROOT/'source-sha256.json')
    for name,value in source_manifest.items():assert sha(ROOT/name)==value
    report_manifest=load(orientation_results/'report-file-sha256.json')
    for name,value in report_manifest.items():assert sha(orientation_results.parent/name)==value
    for name,value in load(orientation_results/'input-bindings.json').items():assert sha(name)==value
    output.mkdir(parents=True,exist_ok=False)
    process=psutil.Process();telemetry=[];start=time.monotonic()
    def status(phase,**extra):
        row=dict(elapsed_seconds=time.monotonic()-start,pid=process.pid,
            rss_MiB=process.memory_info().rss/2**20,
            available_GiB=psutil.virtual_memory().available/2**30,phase=phase,**extra)
        telemetry.append(row);write(output/'status.json',dict(state='RUNNING',**row))
        with (output/'telemetry.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    bindings={};rows=[];omissions=[];timings=[];budgets=[];panels=[]
    maximum_value_error=maximum_jacobian_error=maximum_gradient_error=maximum_update_error=0.
    for dataset in DATASETS:
        bank=orientation/'output/banks'/dataset
        path=bank/'field.npz';bindings[str(path)]=sha(path)
        with np.load(path) as data:x,t,truth,target=[data[k] for k in ('x','t','oracle','pair')]
        for family in FAMILIES:
            for trained,runtime in (('gradient',parent/'output'),('directional',orientation/'output')):
                status('verify_and_evaluate',dataset=dataset,family=family,trained=trained)
                path=runtime/'runs'/dataset/(family+'_'+trained)/'step_0005000.pt'
                bindings[str(path)]=sha(path)
                state=torch.load(path,map_location='cpu',weights_only=False)
                reference,explicit=make_pair(state,x.shape[1],family)
                gradient_error,update_error=gradients_and_update(reference,explicit,x,t,target)
                maximum_gradient_error=max(maximum_gradient_error,gradient_error)
                maximum_update_error=max(maximum_update_error,update_error)
                for model in (reference,explicit):
                    for parameter in model.parameters():parameter.requires_grad_(False)
                expected=field(reference,x,t,torch.float64)
                actual=field(explicit,x,t,torch.float64)
                np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10)
                maximum_value_error=max(maximum_value_error,float(np.max(np.abs(actual-expected))))
                # Compare native float32 values at actual learned parameters too.
                a32,b32=make_pair(state,x.shape[1],family,torch.float32)
                expected32=field(a32,x,t,torch.float32);actual32=field(b32,x,t,torch.float32)
                np.testing.assert_allclose(actual32,expected32,rtol=5e-4,atol=5e-5)
                folder=output/'fields'/dataset/family/trained;folder.mkdir(parents=True)
                np.savez_compressed(folder/'field.npz',exact=actual,reference=expected,
                    explicit_float32=actual32,reference_float32=expected32)
                rows.append(dict(dataset=dataset,family=family,trained=trained,
                    oracle_mse=float(np.mean((actual-truth)**2)),
                    value_error=float(np.max(np.abs(actual-expected))),
                    float32_value_error=float(np.max(np.abs(actual32-expected32))),
                    gradient_error=gradient_error,optimizer_update_error=update_error))
                omitted=None
                if family=='ode_k2' and trained=='gradient':
                    omitted=ExplicitAdjointNet(explicit.raw,include_clock=False).eval()
                    altered=field(omitted,x,t,torch.float64)
                    change=altered-actual
                    alignment=2*np.mean((actual-truth)*change,axis=1)
                    energy=np.mean(change**2,axis=1)
                    base_error=np.mean((actual-truth)**2,axis=1)
                    changed_error=np.mean((altered-truth)**2,axis=1)
                    np.testing.assert_allclose(changed_error-base_error,alignment+energy,rtol=1e-9,atol=1e-10)
                    np.savez_compressed(folder/'clock-omission.npz',exact=actual,altered=altered,
                        exact_error=base_error,altered_error=changed_error,alignment=alignment,change_energy=energy)
                    omissions.append(dict(dataset=dataset,exact_mse=float(base_error.mean()),
                        omitted_clock_mse=float(changed_error.mean()),alignment=float(alignment.mean()),
                        change_energy=float(energy.mean())))
                for clock in TIMES:
                    path=bank/f'panel_{clock:.1f}.npz';bindings[str(path)]=sha(path)
                    with np.load(path) as data:px,pt,pj=[data[k] for k in ('x','t','oracle_jacobian')]
                    ev,ej=chunked_derivative(reference,px,pt)
                    av,aj=chunked_derivative(explicit,px,pt)
                    np.testing.assert_allclose(av,ev,rtol=1e-9,atol=1e-10)
                    np.testing.assert_allclose(aj,ej,rtol=1e-8,atol=1e-9)
                    maximum_jacobian_error=max(maximum_jacobian_error,float(np.max(np.abs(aj-ej))))
                    arrays=dict(prediction=av,jacobian=aj,reference_prediction=ev,reference_jacobian=ej)
                    record=dict(dataset=dataset,family=family,trained=trained,time=clock,
                        exact_jacobian_error=float(np.mean(np.sum((aj-pj)**2,axis=(-1,-2)))),
                        jacobian_formula_error=float(np.max(np.abs(aj-ej))))
                    if omitted is not None:
                        ov,oj=chunked_derivative(omitted,px,pt)
                        arrays.update(omitted_prediction=ov,omitted_jacobian=oj)
                        record['omitted_jacobian_error']=float(np.mean(np.sum((oj-pj)**2,axis=(-1,-2))))
                    panels.append(record);np.savez_compressed(folder/f'panel_{clock:.1f}.npz',**arrays)
                if trained=='gradient':
                    status('benchmark',dataset=dataset,family=family)
                    aa,bb,tt=benchmark(state,x.shape[1],family,x,t,target)
                    for method,budget in (('autograd',aa),('explicit',bb)):
                        budgets.append(dict(dataset=dataset,family=family,method=method,**budget))
                    for record in tt:timings.append(dict(dataset=dataset,family=family,**record))
    for name,value in source_manifest.items():assert sha(ROOT/name)==value
    for name,value in bindings.items():assert sha(name)==value
    table(output/'comparison.csv',rows);table(output/'clock-omission.csv',omissions)
    table(output/'panels.csv',panels);table(output/'budgets.csv',budgets)
    write(output/'timings.json',timings);write(output/'input-bindings.json',bindings)
    audit=dict(status='PASS',complete_fields=len(rows),complete_panels=len(panels),
        all_parameter_gradient_cases=len(rows),actual_optimizer_update_cases=len(rows),
        clock_omission_tasks=len(omissions),benchmark_states=len(budgets)//2,
        maximum_float64_value_error=maximum_value_error,maximum_input_J_error=maximum_jacobian_error,
        maximum_parameter_or_time_gradient_error=maximum_gradient_error,
        maximum_optimizer_update_error=maximum_update_error,
        maximum_RSS_MiB=max(r['rss_MiB'] for r in telemetry),
        minimum_available_GiB=min(r['available_GiB'] for r in telemetry),
        source_manifest_sha256=sha(ROOT/'source-sha256.json'))
    write(output/'audit.json',audit)
    write(output/'status.json',dict(state='COMPLETE',elapsed_seconds=time.monotonic()-start,**audit))
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-runtime',type=Path,required=True)
    p.add_argument('--orientation-runtime',type=Path,required=True)
    p.add_argument('--orientation-results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.parent_runtime.resolve(),a.orientation_runtime.resolve(),
                         a.orientation_results.resolve(),a.output.resolve())
