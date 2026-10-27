"""Independent centered-basis/QR projectors, scalar Heun, and zero-P replay."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=args.root.resolve();out=args.output.resolve();out.mkdir(exist_ok=False);sys.path.insert(0,str(root/'source'))
import invariant_study as formal
started=time.monotonic();formal.verify_inputs(root);settings,base,previous,native,core=formal.setup(root)
assert formal.load(root/'output/status.json')['state']=='COMPLETE'
original={str(p):formal.sha(p) for p in (root/'output').rglob('*') if p.is_file()}
read=lambda name:list(csv.DictReader((root/'output'/name).open()))
rows=read('invariants.csv');raw_rows=read('raw-output-checks.csv');interventions=read('P-interventions.csv');controls=read('dimension-controls.csv')
assert (len(rows),len(raw_rows),len(interventions),len(controls))==(18,90,6,2)
target_basis=np.load(base/'basis.npy');records=[];probes=0;replays=0;max_projector=0.


def readout_columns(state):
    w=state['model']['raw.output.weight'].double().numpy();gamma=state['model']['raw.norm.weight'].double().numpy()
    beta=state['model']['raw.norm.bias'].double().numpy();bias=state['model']['raw.output.bias'].double().numpy()
    width=len(gamma);unit=np.ones(width)/np.sqrt(width);v=np.eye(width)[:,0]-unit
    householder=np.eye(width)-2*np.outer(v,v)/(v@v);z=householder[:,1:]
    np.testing.assert_allclose(z.T@z,np.eye(width-1),atol=1e-12)
    np.testing.assert_allclose(z@z.T,np.eye(width)-np.outer(unit,unit),atol=1e-12)
    return (w*gamma[None])@z,w@beta+bias


def qr_null(columns):
    q,r=np.linalg.qr(columns,mode='reduced');left,sigma,_=np.linalg.svd(r,full_matrices=False)
    rank=int(np.sum(sigma>1e-10*max(float(sigma[0]) if len(sigma) else 0.,1.)))
    span=q@left[:,:rank];projector=np.eye(len(columns))-span@span.T
    np.testing.assert_allclose(columns.T@projector,0,rtol=0,atol=1e-8)
    return projector,rank


def scalar_rule(head,initial,bias):
    value=float(initial);step=1/32
    def velocity(x,t):
        if head=='direct':return bias
        delta=max(1-t,.05)
        return (bias-x)/delta if head=='clean' else bias-x/delta
    for i in range(32):
        first=velocity(value,i*step)
        predicted=value+step*first
        value+=step*(first+velocity(predicted,(i+1)*step))/2
    return value


def compare(source,generated,projector,c0,r,d):
    x=source.astype(np.float64)@projector;y=generated.astype(np.float64)@projector
    expected=x*r+d*(c0@projector);error=float(np.max(np.abs(y-expected)));assert error<=1e-4,error
    x-=x.mean(0);y-=y.mean(0)
    covariance=y.T@y/(len(y)-1);prediction=(x.T@x)*r*r/(len(x)-1)
    relative=float(np.linalg.norm(covariance-prediction)/max(np.linalg.norm(prediction),1e-30));assert relative<=1e-4
    return error,relative


for row in rows:
    law,family,head=[row[k] for k in ('law','family','head')];arm=f'{family}_{head}';folder=base/'output/runs'/law/arm
    state=torch.load(folder/'step_0005000.pt',weights_only=False);columns,c0=readout_columns(state)
    projector,rank=qr_null(columns);normal,combined_rank=qr_null(np.column_stack((columns,target_basis)))
    assert rank==int(row['readout_rank']) and 128-combined_rank==int(row['normal_null_dimension'])
    with np.load(root/'output/raw'/f'{law}-{arm}.npz') as data:
        discrepancy=max(float(np.max(np.abs(projector-data['projector']))),float(np.max(np.abs(normal-data['normal_projector']))))
        np.testing.assert_allclose(projector,data['projector'],rtol=0,atol=1e-8)
        np.testing.assert_allclose(normal,data['normal_projector'],rtol=0,atol=1e-8)
        max_projector=max(max_projector,discrepancy)
    r,d=scalar_rule(head,1.,0.),scalar_rule(head,0.,1.)
    np.testing.assert_allclose([r,d],[float(row['Heun_R']),float(row['Heun_d'])],rtol=1e-12,atol=1e-12)
    banks=base/'banks'/law
    with np.load(banks/'sampling.npz') as data:source=data['source']
    with np.load(folder/'diagnostics/samples.npz') as data:generated=data['generated']
    error,covariance=compare(source,generated,projector,c0,r,d);normal_error,normal_covariance=compare(source,generated,normal,c0,r,d)
    np.testing.assert_allclose(error,float(row['identity_max']),rtol=1e-5,atol=1e-9)
    variance=float(row['target_normal_variance']);normal_dim=128-combined_rank
    mean_energy=d*d*float(np.sum((c0@normal)**2));w2=mean_energy+normal_dim*(abs(r)-np.sqrt(variance))**2
    np.testing.assert_allclose(w2,float(row['population_full_W2_squared_lower_bound']),rtol=1e-8,atol=1e-10)
    model=previous.build(base,law,arm);native.restore_model(model,state);model.eval()
    for clock in previous.TIMES:
        with np.load(banks/f'panel_{clock:.1f}.npz') as data:x,t=data['x'],data['t']
        with torch.no_grad():raw=model.raw(torch.from_numpy(x),torch.from_numpy(t)).double().numpy()
        maximum=float(np.max(np.abs(raw@projector-c0@projector)));assert maximum<=2e-5
        old=next(v for v in raw_rows if v['law']==law and v['family']==family and v['head']==head and abs(float(v['time'])-clock)<1e-7)
        np.testing.assert_allclose(maximum,float(old['raw_projection_max']),rtol=1e-5,atol=1e-9);probes+=1
    if family=='ode_k2':
        names=[]
        with torch.no_grad():
            for name,layer in model.ode_layers():layer.proj.weight.fill_(0);names.append('raw.'+name+'.proj.weight')
        assert len(names)==6
        for name,value in model.state_dict().items():
            if name in names:assert torch.count_nonzero(value)==0
            else:assert torch.equal(value,state['model'][name]),name
        for name,value in model.named_buffers():assert torch.equal(value,state['runtime_buffers'][name]),name
        rerun=native.legacy.generated_samples(model,torch.from_numpy(source),32).numpy()
        with np.load(root/'output/raw'/f'{law}-{arm}-zero-P.npz') as data:np.testing.assert_array_equal(rerun,data['generated'])
        zero_error,_=compare(source,rerun,projector,c0,r,d)
        zero_normal,_=compare(source,rerun,normal,c0,r,d)
        old=next(v for v in interventions if v['law']==law and v['head']==head)
        difference=rerun.astype(np.float64)-generated;displacement=float(np.sqrt(np.mean(difference**2)))
        np.testing.assert_allclose(displacement,float(old['whole_sample_displacement_rms']),rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(zero_error,float(old['identity_max']),rtol=1e-5,atol=1e-9);replays+=1
    records.append(dict(law=law,family=family,head=head,projector_max_difference=discrepancy,source_identity_max=error,
        covariance_relative_frobenius=covariance,normal_identity_max=normal_error))
    print(dict(law=law,arm=arm,projector_difference=discrepancy,elapsed=time.monotonic()-started),flush=True)

for entry in settings['controls']:
    state=torch.load(entry['checkpoint'],weights_only=False);columns,_=readout_columns(state);p,rank=qr_null(columns)
    old=next(r for r in controls if r['law']==entry['law'])
    assert rank==int(old['readout_rank'])==entry['dim'];assert np.linalg.norm(p)<1e-8
assert probes==90 and replays==6
formal.verify_inputs(root)
for path,digest in original.items():assert formal.sha(Path(path))==digest,path
formal.table(out/'independent-checks.csv',records)
formal.write(out/'audit.json',dict(status='PASS',models=18,raw_probe_panels=probes,zero_P_sampler_replays=replays,dimension_controls=2,
    projector_max_difference=max_projector,formal_files_unchanged=len(original),elapsed_seconds=time.monotonic()-started,
    scope='independent centered-basis/QR subspaces, scalar integration and native counterfactual-sampler replay; no new training'))
for name in ('invariants.csv','raw-output-checks.csv','P-interventions.csv','dimension-controls.csv','receipt.json','telemetry.csv'):shutil.copyfile(root/'output'/name,out/name)
shutil.copyfile(root/'preflight.json',out/'preflight.json')
receipt=formal.load(root/'output/receipt.json');commit=settings['registration_commit'][:7]
with (root/'results.tsv').open('w') as stream:
    stream.write('commit\tmetric\tmemory_gb\tstatus\tdescription\n')
    for row in rows:stream.write(f"{commit}\t{float(row['identity_max']):.10g}\t{receipt['peak_sampled_rss_gib']:.3f}\tkeep\tread-only source identity {row['law']} {row['family']} {row['head']}; no nomination\n")
    for row in interventions:stream.write(f"{commit}\t{float(row['identity_max']):.10g}\t{receipt['peak_sampled_rss_gib']:.3f}\tkeep\tzero-P invariant check {row['law']} {row['head']}; no quality selection\n")
shutil.copyfile(root/'results.tsv',out/'results.tsv');print(dict(status='AUDIT_PASS',projector_max_difference=max_projector,elapsed=time.monotonic()-started),flush=True)
