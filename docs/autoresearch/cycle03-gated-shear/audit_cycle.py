"""Replay native evidence and inspect literal generators on real FFN inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
root=args.root.resolve();out=args.output.resolve();out.mkdir(exist_ok=False);sys.path.insert(0,str(root/'source'))
import study
settings,native,component=study.setup(root);started=time.monotonic()
assert study.load(root/'output/status.json')['state']=='COMPLETE';study.verify_inputs(root)
formal={str(path):study.sha(path) for path in (root/'output').rglob('*') if path.is_file()}
counts=dict(checkpoints=0,panels=0,sampling=0,literal_ffns=0,literal_gradient_probes=0,paired_snapshots=0,operation_traces=0)
stats=[];gate_arrays={};literal_value_max=literal_gradient_max=0.


class Trace(TorchDispatchMode):
    def __init__(self):super().__init__();self.operations=Counter()
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        self.operations[str(func)]+=1
        return func(*args,**(kwargs or {}))


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


for entry in settings['cohort']:
    folder,banks=Path(entry['folder']),Path(entry['banks'])
    for step in (0,1000,2500,5000):
        model=study.build(native,entry);state=torch.load(folder/f'step_{step:07d}.pt',weights_only=False)
        native.restore_model(model,state);shear=study.converted(model,component)
        tag=f"{entry['law']}-{entry['family']}-{step}"
        with np.load(banks/'field.npz') as data:x,t=data['x'],data['t']
        with np.load(root/'output/predictions'/(tag+'.npz')) as data:saved=data['prediction']
        with torch.no_grad():
            for k in range(0,len(x),1024):
                xx,tt=torch.from_numpy(x[k:k+1024]),torch.from_numpy(t[k:k+1024])
                a,b=model(xx,tt),shear(xx,tt);assert torch.equal(a,b)
                np.testing.assert_array_equal(a.numpy(),saved[k:k+1024])
        visits=[];hooks=[]
        for index,block in enumerate(model.raw.blocks):
            hooks.append(block.ffn.register_forward_pre_hook(lambda module,args,index=index:visits.append((index,module,args[0].detach().clone()))))
        with np.load(banks/'probe.npz') as data:
            with torch.no_grad():model(torch.from_numpy(data['x']),torch.from_numpy(data['t']))
        for hook in hooks:hook.remove()
        assert len(visits)==3
        for index,ffn,visited in visits:
            with torch.no_grad():
                gate=F.silu(ffn.w12(visited).chunk(2,-1)[0]);absolute=gate.abs().double();maximum=absolute.amax(-1)
                flow=.5*(torch.sqrt(4+maximum.square())+maximum)
                stats.append(dict(law=entry['law'],family=entry['family'],step=step,layer=f'block{index}',
                    sigma_q0=float(absolute.min()),sigma_q25=float(torch.quantile(absolute,.25)),
                    sigma_q50=float(torch.quantile(absolute,.5)),sigma_q75=float(torch.quantile(absolute,.75)),
                    sigma_q95=float(torch.quantile(absolute,.95)),sigma_q100=float(absolute.max()),
                    flow_norm_mean=float(flow.mean()),flow_norm_max=float(flow.max()),generator_eigenvalue=0.))
                gate_arrays[tag+f'-block{index}']=absolute.numpy()
            # Literal dense I+P at one actual visited state, with all encoder derivatives live.
            local=visited[:1].clone().requires_grad_(True);a,v=ffn.w12(local).chunk(2,-1);g=F.silu(a)
            diagonal=torch.diag_embed(g);zero=torch.zeros_like(diagonal);matrix=torch.cat([torch.cat([zero,diagonal],-1),torch.cat([zero,zero],-1)],-2)
            assert torch.equal(matrix@matrix,torch.zeros_like(matrix))
            initial=torch.cat([torch.zeros_like(v),v],-1)
            endpoint=initial+torch.bmm(matrix,initial.unsqueeze(-1)).squeeze(-1)
            literal=ffn.w3(endpoint[:,:v.shape[-1]]);ordinary=ffn(local)
            torch.testing.assert_close(literal,ordinary,rtol=2e-5,atol=3e-6)
            literal_value_max=max(literal_value_max,float((literal-ordinary).abs().max().detach()))
            sv=torch.linalg.svdvals(matrix.double())[0]
            expected=torch.sort(torch.cat([g.abs(),torch.zeros_like(g)],-1).double(),descending=True).values[0]
            torch.testing.assert_close(sv,expected,rtol=1e-10,atol=1e-11)
            counts['literal_ffns']+=1
            if step==5000:
                coefficient=torch.cos(torch.arange(literal.numel(),dtype=literal.dtype)).reshape_as(literal)
                derivatives=[torch.autograd.grad((result*coefficient).sum(),(local,*ffn.parameters()),retain_graph=True) for result in (literal,ordinary)]
                for aa,bb in zip(*derivatives):
                    torch.testing.assert_close(aa,bb,rtol=2e-5,atol=3e-6)
                    literal_gradient_max=max(literal_gradient_max,float((aa-bb).abs().max()))
                counts['literal_gradient_probes']+=1
        counts['checkpoints']+=1
        if step==5000:
            traces=[]
            for m in (model,shear):
                with Trace() as trace:
                    with torch.no_grad():m(torch.from_numpy(x[:4]),torch.from_numpy(t[:4]))
                traces.append(dict(trace.operations))
            assert traces[0]==traces[1];counts['operation_traces']+=1
            for clock in (.1,.3,.5,.7,.9):
                with np.load(banks/f'panel_{clock:.1f}.npz') as data:xx,tt=torch.from_numpy(data['x']),torch.from_numpy(data['t'])
                a,ja=native.legacy.jacobian(model,xx,tt);b,jb=native.legacy.jacobian(shear,xx,tt)
                assert torch.equal(a,b) and torch.equal(ja,jb);counts['panels']+=1
            with np.load(banks/'sampling.npz') as data:source=torch.from_numpy(data['source'])
            samples=native.legacy.generated_samples(shear,source,32)
            with np.load(folder/'diagnostics/samples.npz') as data:np.testing.assert_array_equal(samples.numpy(),data['generated'])
            counts['sampling']+=1
        print(dict(law=entry['law'],family=entry['family'],step=step,elapsed=time.monotonic()-started),flush=True)

entry=next(r for r in settings['cohort'] if r['law']=='half_mixture' and r['family']=='swiglu128')
for step in (0,1000,2500,5000):
    actual=torch.load(root/'output/paired-training'/f'step_{step:07d}.pt',weights_only=False)
    expected=torch.load(Path(entry['folder'])/f'step_{step:07d}.pt',weights_only=False)
    equal(actual,expected);counts['paired_snapshots']+=1
assert counts==dict(checkpoints=44,panels=55,sampling=11,literal_ffns=132,literal_gradient_probes=33,paired_snapshots=4,operation_traces=11)
study.verify_inputs(root)
for path,digest in formal.items():assert study.sha(Path(path))==digest,path
study.table(out/'generator-quantiles.csv',stats);np.savez_compressed(out/'raw-generator-singular-values.npz',**gate_arrays)
for name in ('compatibility.csv','generator-stats.csv','costs.csv','paired-training.csv','decision.json'):
    shutil.copyfile(root/'output'/name,out/name)
shutil.copyfile(root/'preflight.json',out/'preflight.json')
telemetry=[json.loads(line) for line in (root/'output/monitor.jsonl').read_text().splitlines()]
study.write(out/'audit.json',dict(status='PASS',counts=counts,literal_value_max=literal_value_max,
    literal_gradient_max=literal_gradient_max,formal_files_unchanged=len(formal),bindings=len(study.load(root/'bindings.json')),
    elapsed_seconds=time.monotonic()-started))
study.write(out/'execution.json',dict(formal=study.load(root/'output/status.json'),
    peak_sampled_rss_gib=max(row['rss_bytes'] for row in telemetry)/2**30,telemetry_samples=len(telemetry),
    registration_commit=settings['registration_commit'],quality_training_runs=0,paired_compatibility_trajectories=1))
print(dict(status='AUDIT_PASS',counts=counts,elapsed=time.monotonic()-started),flush=True)
