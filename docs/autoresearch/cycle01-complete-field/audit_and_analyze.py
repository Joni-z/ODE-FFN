"""Independent automatic-differentiation replay and descriptive error analysis."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=args.root.resolve();out=args.output.resolve();out.mkdir(exist_ok=False)
sys.path.insert(0,str(root/'source'))
import run_cycle as study
settings,build,compute_budget,restore,legacy=study.setup(root)
assert study.load(root/'output/status.json')['state']=='COMPLETE'
started=time.monotonic();study.verify_bindings(root)
formal={str(p):study.sha(p) for p in (root/'output').rglob('*') if p.is_file()}


def manual_norm(layer,h):
    centered=h-h.mean(-1,keepdim=True)
    return centered*torch.rsqrt(centered.square().mean(-1,keepdim=True)+layer.eps)*layer.weight+layer.bias


class Independent(nn.Module):
    """AD tangents, independently expressed finite updates, unchanged native weights."""
    def __init__(self,parent,method,position):
        super().__init__();self.raw=parent.raw;self.dim=parent.dim;self.method=method;self.position=position

    def forward(self,x,t,*,updates=3,step_size=None):
        assert updates==3 and step_size is None
        raw=self.raw;emb=raw.time(t);h=raw.input(x)
        for index,b in enumerate(raw.blocks):
            if self.method=='euler' or self.position not in ('all',str(index+1)):
                h=b(h,emb);continue
            scale,shift=b.condition(emb).chunk(2,-1)
            def f(state):
                return b.ffn(manual_norm(b.norm,state)*(1+scale)+shift)
            first=f(h);c=b.residual_scale
            if self.method in ('taylor','frozen_gate'):
                if self.method=='frozen_gate':
                    fixed=b.ffn.w12(manual_norm(b.norm,h)*(1+scale)+shift).chunk(2,-1)[0]
                    def function(state):
                        value=b.ffn.w12(manual_norm(b.norm,state)*(1+scale)+shift).chunk(2,-1)[1]
                        return b.ffn.w3(F.silu(fixed)*value)
                else:function=f
                _,tangent=torch.autograd.functional.jvp(function,h,first,create_graph=torch.is_grad_enabled())
                h=h+c*first+(c*c/2)*tangent
            elif self.method=='midpoint':
                h=h+c*f(h+(c/2)*first)
            else:
                # Independent sum-of-velocities form of two half-Euler steps.
                h=h+(c/2)*(first+f(h+(c/2)*first))
        return raw.output(raw.norm(h))


def csv_rows(name):
    return list(csv.DictReader((root/'output'/name).open()))


def close(a,b,atol=1e-5):
    np.testing.assert_allclose(a,b,rtol=2e-5,atol=atol)


rows=csv_rows('field.csv');panels=csv_rows('panels.csv');samples=csv_rows('sampling.csv');costs=csv_rows('costs.csv')
counts=dict(field_cases=0,prediction_coordinates=0,panels=0,sampling=0)
decomposition=[];examples=[];sampling_differences=[];largest_prediction_difference=0.
for entry in settings['cohort']:
    dataset=entry['dataset'];banks=Path(entry['banks'])
    with np.load(banks/'field.npz') as data:x,t,oracle,pair=[data[k] for k in ('x','t','oracle','pair')]
    expected_count=10000 if entry['dim']==2 else 20000
    assert len(x)==expected_count,(dataset,len(x),expected_count)
    for step in study.STEPS:
        parent=study.restored(build,restore,entry,step)
        basetag=f'{dataset}-{step}-euler-all'
        with np.load(root/'output/predictions'/(basetag+'.npz')) as data:baseline=data['prediction'].astype(np.float64)
        for row in [r for r in rows if r['dataset']==dataset and int(r['step'])==step]:
            method,position=row['method'],row['position'];model=Independent(parent,method,position)
            tag=f'{dataset}-{step}-{method}-{position}'
            with torch.no_grad():
                prediction=np.concatenate([model(torch.from_numpy(x[k:k+1024]),torch.from_numpy(t[k:k+1024])).numpy() for k in range(0,len(x),1024)])
            with np.load(root/'output/predictions'/(tag+'.npz')) as data:saved=data['prediction']
            close(prediction,saved)
            largest_prediction_difference=max(largest_prediction_difference,float(np.max(np.abs(prediction-saved))))
            close(float(np.mean((saved-oracle)**2)),float(row['oracle_mse']),atol=1e-8)
            close(float(np.mean((saved-pair)**2)),float(row['pair_mse']),atol=1e-8)
            counts['field_cases']+=1;counts['prediction_coordinates']+=saved.size
            error=baseline-oracle.astype(np.float64);delta=saved.astype(np.float64)-baseline
            linear=float(2*np.mean(error*delta));energy=float(np.mean(delta**2))
            total=float(np.mean((error+delta)**2)-np.mean(error**2))
            np.testing.assert_allclose(linear+energy,total,rtol=1e-10,atol=1e-12)
            decomposition.append(dict(dataset=dataset,step=step,method=method,position=position,
                cross_error_term=linear,displacement_energy=energy,total_mse_change=total,
                improving_point_fraction=float(np.mean(np.mean((error+delta)**2,axis=1)<np.mean(error**2,axis=1)))))
            if step==5000 and method=='taylor' and position=='all':
                change=np.mean((error+delta)**2,axis=1)-np.mean(error**2,axis=1)
                for label,index in [('largest_increase',int(np.argmax(change))),('largest_decrease',int(np.argmin(change))),
                                    ('median_change',int(np.argsort(change,kind='stable')[len(change)//2]))]:
                    examples.append(dict(dataset=dataset,selection=label,index=index,time=float(t[index,0]),
                        x=x[index].tolist(),oracle=oracle[index].tolist(),euler=baseline[index].tolist(),
                        taylor=saved[index].tolist(),mse_change=float(change[index])))
            if step==5000 and position=='all':
                for clock in study.TIMES:
                    with np.load(banks/f'panel_{clock:.1f}.npz') as data:
                        xx,tt,truth,jj=[data[k] for k in ('x','t','oracle','oracle_jacobian')]
                    value,jac=legacy.jacobian(model,torch.from_numpy(xx),torch.from_numpy(tt))
                    with np.load(root/'output/predictions'/f'{tag}-panel-{clock:.1f}.npz') as data:
                        close(value.numpy(),data['prediction'],atol=1e-5)
                        close(jac.numpy(),data['jacobian'],atol=3e-5)
                    counts['panels']+=1
                with np.load(banks/'sampling.npz') as data:
                    source,target,directions=[torch.from_numpy(data[k]) for k in ('source','target','directions')]
                generated=legacy.generated_samples(model,source,32)
                with np.load(root/'output/predictions'/f'{tag}-samples.npz') as data:
                    difference=generated.numpy().astype(np.float64)-data['generated'].astype(np.float64)
                    sample_max=float(np.max(np.abs(difference)));sample_rms=float(np.sqrt(np.mean(difference**2)))
                    assert sample_max<1e-3 and sample_rms<1e-5,(tag,sample_max,sample_rms)
                    sampling_differences.append(dict(dataset=dataset,method=method,max_absolute_difference=sample_max,rms_difference=sample_rms))
                sample=next(s for s in samples if s['dataset']==dataset and s['method']==method)
                close(legacy.sliced_w2(generated,target,directions),float(sample['sw2']),atol=2e-6)
                counts['sampling']+=1
        print(dict(dataset=dataset,step=step,audited=counts['field_cases'],elapsed=time.monotonic()-started),flush=True)
assert counts['field_cases']==204 and counts['panels']==100 and counts['sampling']==20
study.verify_bindings(root)
for path,digest in formal.items():assert study.sha(path)==digest,path
study.write(out/'audit.json',dict(status='PASS',counts=counts,bindings=91,
    formal_outputs_unchanged=len(formal),max_prediction_difference=largest_prediction_difference,
    elapsed_seconds=time.monotonic()-started))
study.table(out/'error-decomposition.csv',decomposition)
study.table(out/'sampling-replay-differences.csv',sampling_differences)
study.write(out/'examples.json',examples)

# Registered endpoint only; positions and historical snapshots cannot nominate substitutes.
endpoint=[r for r in rows if int(r['step'])==5000 and r['position']=='all']
comparison=[]
for dataset in dict.fromkeys(r['dataset'] for r in endpoint):
    base=next(r for r in endpoint if r['dataset']==dataset and r['method']=='euler')
    base_j=np.mean([float(p['jacobian_error']) for p in panels if p['dataset']==dataset and p['method']=='euler'])
    base_s=float(next(s['sw2'] for s in samples if s['dataset']==dataset and s['method']=='euler'))
    for row in [r for r in endpoint if r['dataset']==dataset]:
        method=row['method'];mse=float(row['oracle_mse'])
        jac=np.mean([float(p['jacobian_error']) for p in panels if p['dataset']==dataset and p['method']==method])
        sw2=float(next(s['sw2'] for s in samples if s['dataset']==dataset and s['method']==method))
        comparison.append(dict(dataset=dataset,method=method,oracle_mse=mse,mse_ratio=mse/float(base['oracle_mse']),
            jacobian_error=float(jac),jacobian_ratio=float(jac/base_j),sw2=sw2,sw2_ratio=sw2/base_s))
study.table(out/'comparison.csv',comparison)
gates=[]
for dataset in dict.fromkeys(r['dataset'] for r in endpoint):
    values=[r for r in comparison if r['dataset']==dataset]
    candidate=next(r for r in values if r['method']=='taylor')
    nonlinear=dataset!='gaussian_aniso_2d';dependent=dataset in ('rotated_product_bimodal_16d','sign_code')
    gates.append(dict(dataset=dataset,field_pass=candidate['mse_ratio']<=(.95 if dependent else 1.05),
        jacobian_pass=not nonlinear or candidate['jacobian_ratio']<=1.10,
        sampling_pass=not nonlinear or candidate['sw2_ratio']<=1.10,
        compute_control_pass=not dependent or candidate['oracle_mse']<min(r['oracle_mse'] for r in values if r['method'] in ('midpoint','half_euler'))))
passed=all(all(v for k,v in gate.items() if k!='dataset') for gate in gates)
study.write(out/'decision.json',dict(status='keep_for_training_review' if passed else 'discard',gates=gates,
    hypothesis='post-training complete-Taylor upgrade at the native residual duration',
    claim_scope='Frozen seed0 SwiGLU128 checkpoints on these four validation laws; not a rejection of fresh higher-order training.',
    next_training_authorized_by_this_cycle=passed))
revision=study.load(root/'settings.json')['base_commit']
with (root/'results.tsv').open('w') as stream:
    stream.write('commit\tmetric\tmemory_gb\tstatus\tdescription\n')
    telemetry=[json.loads(line) for line in (root/'output/monitor.jsonl').read_text().splitlines()]
    peak=max(v['rss_bytes'] for v in telemetry)/2**30
    for row in comparison:
        status='keep' if row['method']=='euler' else 'discard'
        stream.write(f"e0ad302\t{row['oracle_mse']:.10f}\t{peak:.3f}\t{status}\t{row['dataset']} {row['method']} frozen checkpoint\n")
shutil_names=('field.csv','panels.csv','sampling.csv','costs.csv')
import shutil
for name in shutil_names:shutil.copyfile(root/'output'/name,out/name)
shutil.copyfile(root/'preflight.json',out/'preflight.json')
shutil.copyfile(root/'results.tsv',out/'results.tsv')
study.write(out/'execution.json',dict(registration_commit='e0ad302',formal=study.load(root/'output/status.json'),
    model_seed=0,new_training_runs=0,telemetry_samples=len(telemetry),peak_sampled_rss_gib=peak,
    immutable_input_sha256=study.sha(root/'bindings.json'),upstream_revision=settings['upstream_revision']))
print(dict(status='AUDIT_PASS',decision='keep' if passed else 'discard',elapsed=time.monotonic()-started),flush=True)
