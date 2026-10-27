"""Independent data/oracle/trajectory replay for the fixed-component-count control."""
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
args=parser.parse_args();root=args.root.resolve();out=args.output.resolve();out.mkdir(exist_ok=False)
sys.path.insert(0,str(root/'source'))
import study
settings,ref=study.setup(root);started=time.monotonic();assert study.load(root/'output/status.json')['state']=='COMPLETE'
study.verify(root,ref);law=study.target(root);r=torch.tensor(law.rotation);mu=torch.tensor(law.locations);means=torch.tensor(law.means)
formal={str(p):ref.sha(p) for p in (root/'output').rglob('*') if p.is_file()}


def independent(x,t):
    # Full residual Gaussian likelihoods; no equal-norm/partition simplification.
    q=x@r;d=(1-t)**2+.04*t**2
    pair=torch.stack([mu,-mu],dim=-1)
    coordinate_log=-.5*(q[:,:,None]-t[:,:,None]*pair[None]).square()/d[:,:,None]
    coordinate_p=torch.softmax(coordinate_log,dim=-1)
    mean0=(coordinate_p*pair).sum(-1)
    log0=(torch.logsumexp(coordinate_log,dim=-1)-np.log(2)).sum(-1,keepdim=True)
    code_log=-.5*((q[:,None]-t[:,:,None]*means[None]).square()/d[:,:,None]).sum(-1)
    mean1=torch.softmax(code_log,dim=-1)@means
    log1=torch.logsumexp(code_log,dim=-1,keepdim=True)-np.log(32)
    w=torch.softmax(torch.cat([log0,log1],dim=-1),dim=-1)[:,1:]
    mean=(1-w)*mean0+w*mean1
    return (((.04*t-(1-t))/d)*q+(1-t)/d*mean)@r.T


def close(a,b):np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-7)


# Independent regeneration uses explicit branch, sign, Gaussian and rotation operations.
rng=np.random.default_rng(0)
def sample(count):
    branch=rng.random(count)<.5
    signs=2*rng.integers(0,2,size=(count,16))-1
    code=law.signs[rng.integers(32,size=count)]
    final=np.where(branch[:,None],code,signs)
    return ((final*law.locations+.2*rng.standard_normal((count,16)))@law.rotation.T).astype(np.float32)
train,validation,probe_target=sample(100000),sample(20000),sample(2048)
np.testing.assert_array_equal(train,np.load(root/'data/train.npy'))
np.testing.assert_array_equal(validation,np.load(root/'data/validation.npy'))
noise=rng.standard_normal((2048,16)).astype(np.float32);tt=rng.random((2048,1)).astype(np.float32)
with np.load(root/'banks/probe.npz') as data:
    np.testing.assert_array_equal(data['x'],((1-tt)*noise+tt*probe_target)[:256]);np.testing.assert_array_equal(data['t'],tt[:256])
with np.load(root/'banks/field.npz') as data:x,t,oracle,pair=[data[k] for k in ('x','t','oracle','pair')]
assert len(x)==20000
with torch.no_grad():
    checked=np.concatenate([independent(torch.tensor(x[k:k+1024],dtype=torch.float64),torch.tensor(t[k:k+1024],dtype=torch.float64)).numpy() for k in range(0,len(x),1024)])
np.testing.assert_allclose(checked,oracle,rtol=1e-6,atol=2e-6)
geometry=[]
for clock in study.TIMES:
    with np.load(root/'banks'/f'panel_{clock:.1f}.npz') as data:
        xx,tt,truth,jac,w=[data[k] for k in ('x','t','oracle','oracle_jacobian','posterior_code_weight')]
    state=torch.tensor(xx,dtype=torch.float64,requires_grad=True);time_tensor=torch.tensor(tt,dtype=torch.float64)
    value=independent(state,time_tensor)
    derivative=torch.stack([torch.autograd.grad(value[:,j].sum(),state,retain_graph=True)[0] for j in range(16)],1)
    np.testing.assert_allclose(value.detach().numpy(),truth,rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(derivative.detach().numpy(),jac,rtol=1e-8,atol=1e-9)
    from joint_core import target_geometry
    shape=target_geometry(jac,law.rotation)
    entropy=-(w*np.log(np.maximum(w,1e-300))+(1-w)*np.log(np.maximum(1-w,1e-300)))
    geometry.append(dict(time=clock,off_diagonal_energy_fraction=float(shape['off_diagonal_energy_fraction'].mean()),
        commutator=float(shape['normalized_pair_commutator'].mean()),posterior_code_weight_mean=float(w.mean()),
        posterior_branch_entropy_nats=float(entropy.mean())))

# Every family consumes the same batch stream. Reconstruct it once, check all snapshots.
bank=torch.from_numpy(train);generator=torch.Generator().manual_seed(100000);digest=ref.EMPTY;digests={0:digest}
for step in range(1,5001):
    a=bank[torch.randint(len(bank),(512,),generator=generator)]
    b=torch.randn(a.shape,generator=generator);tt=torch.rand((512,1),generator=generator)
    digest=ref.legacy.fingerprint_batch(digest,a,b,tt)
    if step in study.SNAPSHOTS:digests[step]=digest
counts=dict(checkpoints=0,panels=0,sampling=0,matrices=0,stream_batches=5000,field_oracle_rows=20000)
curves=[];matrix_rows=[];spectra=[];endpoints=[];training=[]
for family in study.FAMILIES:
    folder=root/'output/runs'/family;assert study.load(folder/'status.json')['state']=='COMPLETE'
    records=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
    for step in study.SNAPSHOTS:
        model=ref.build(family);state=torch.load(folder/f'step_{step:07d}.pt',map_location='cpu',weights_only=False)
        ref.restore_model(model,state);assert state['input_stream_digest']==digests[step]
        actual=ref.legacy.field_evaluation(model,root/'banks')
        record=next(r for r in records if r['step']==step)
        for key in ('oracle_mse','pair_mse','prediction_rms'):close(actual[key],record[key])
        with torch.no_grad():prediction=np.concatenate([model(torch.from_numpy(x[k:k+1024]),torch.from_numpy(t[k:k+1024])).numpy() for k in range(0,len(x),1024)])
        with np.load(folder/f'field_{step:07d}.npz') as data:np.testing.assert_array_equal(prediction,data['prediction'])
        for item in study.load(folder/f'matrices_{step:07d}.json'):
            p=state['model'][item['layer']].double().numpy();singular=np.linalg.svd(p,compute_uv=False);eig=np.linalg.eigvals(p)
            np.testing.assert_allclose(singular,item['singular_values'],rtol=1e-9,atol=1e-10)
            recorded=np.array(item['eigenvalues_real'])+1j*np.array(item['eigenvalues_imag'])
            np.testing.assert_allclose(np.min(np.abs(eig[:,None]-recorded[None,:]),axis=1),0,atol=1e-9)
            matrix_rows.append(dict(family=family,step=step,layer=item['layer'],kind=item['kind'],
                sigma_max=float(singular[0]),sigma_min=float(singular[-1]),spectral_radius=float(np.abs(eig).max()),
                max_real_eigenvalue=float(eig.real.max())))
            for index,sigma in enumerate(singular):spectra.append(dict(family=family,step=step,layer=item['layer'],index=index,singular_value=float(sigma)))
            counts['matrices']+=1
        curves.append(dict(family=family,step=step,**actual));counts['checkpoints']+=1
    result=study.load(folder/'result.json');assert result['law']=='full_component_support_half_mixture'
    assert result['input_stream_digest']==digests[5000]
    for key in ('oracle_mse','pair_mse','prediction_rms'):close(actual[key],result[key])
    for clock in study.TIMES:
        with np.load(root/'banks'/f'panel_{clock:.1f}.npz') as data:xx,tt,truth,jj=[data[k] for k in ('x','t','oracle','oracle_jacobian')]
        value,jac=ref.legacy.jacobian(model,torch.from_numpy(xx),torch.from_numpy(tt))
        with np.load(folder/'diagnostics'/f'panel_{clock:.1f}.npz') as data:
            np.testing.assert_array_equal(value.numpy(),data['prediction']);np.testing.assert_array_equal(jac.numpy(),data['jacobian'])
        row=next(r for r in result['panels'] if abs(r['time']-clock)<1e-6)
        close(float(np.mean((value.numpy()-truth)**2)),row['field_mse'])
        close(float(np.mean(np.sum((jac.numpy()-jj)**2,axis=(1,2)))),row['jacobian_error']);counts['panels']+=1
    with np.load(root/'banks/sampling.npz') as data:source,target,directions=[torch.from_numpy(data[k]) for k in ('source','target','directions')]
    generated=ref.legacy.generated_samples(model,source,32)
    with np.load(folder/'diagnostics/samples.npz') as data:np.testing.assert_array_equal(generated.numpy(),data['generated'])
    close(ref.legacy.sliced_w2(generated,target,directions),result['sw2']);counts['sampling']+=1
    endpoints.append(dict(law='half_mixture',family=family,reused=False,**{k:result[k] for k in ('oracle_mse','jacobian_error','sw2','parameters','dense_forward_macs','dense_training_forward_backward_macs')}))
    for row in records:training.append(dict(family=family,**row))
    print(dict(family=family,audited=counts,elapsed=time.monotonic()-started),flush=True)

for lawname,entries in [('product',ref.reference_rows(root)),('sign_code',[dict(family=f,folder=str(Path(settings['old_joint_root'])/'output/runs'/f)) for f in study.FAMILIES])]:
    for entry in entries:
        result=study.load(Path(entry['folder'])/'result.json')
        endpoints.append(dict(law=lawname,family=entry['family'],reused=True,**{k:result[k] for k in ('oracle_mse','jacobian_error','sw2','parameters','dense_forward_macs','dense_training_forward_backward_macs')}))
assert counts['checkpoints']==20 and counts['panels']==25 and counts['sampling']==5 and counts['matrices']==48
study.verify(root,ref)
for path,digest in formal.items():assert ref.sha(path)==digest,path

def table(name,rows):
    with (out/name).open('w',newline='') as stream:
        keys=list(dict.fromkeys(k for row in rows for k in row));writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
for name,rows in [('comparison.csv',endpoints),('curves.csv',curves),('training.csv',training),('matrix-trajectories.csv',matrix_rows),('matrix-spectra.csv',spectra),('target-geometry.csv',geometry)]:table(name,rows)
mixed=[r for r in endpoints if r['law']=='half_mixture'];candidate=next(r for r in mixed if r['family']=='free_isru');swi=[r for r in mixed if r['family'].startswith('swiglu')]
ratios={key:candidate[key]/min(row[key] for row in swi) for key in ('oracle_mse','jacobian_error','sw2')}
count_insufficient=ratios['oracle_mse']>1.05 or ratios['jacobian_error']>1.10
survives=ratios['oracle_mse']<=.9 and ratios['jacobian_error']<=.9 and ratios['sw2']<=1.10
decision=dict(component_count_alone_insufficient=count_insufficient,ISRU_advantage_survives=survives,ratios=ratios,
    interpretation='Neither outcome identifies coupling as the sole cause; probabilities and effective diversity differ. No new ODE architecture is nominated.')
study.write(out/'decision.json',decision)
study.write(out/'audit.json',dict(status='PASS',counts=counts,bindings=len(study.load(root/'bindings.json')),
    formal_files_unchanged=len(formal),elapsed_seconds=time.monotonic()-started))
study.write(out/'data-summary.json',study.load(root/'data-manifest.json'))
telemetry=[json.loads(line) for line in (root/'output/monitor.jsonl').read_text().splitlines()]
peak=max(sum(p['rss_bytes'] for p in row['processes']) for row in telemetry)/2**30
study.write(out/'execution.json',dict(formal=study.load(root/'output/status.json'),new_training_runs=5,model_seed=0,
    peak_sampled_summed_rss_gib=peak,monitor_samples=len(telemetry),registration_commit=settings['registration_commit']))
with (root/'results.tsv').open('w') as stream:
    stream.write('commit\tmetric\tmemory_gb\tstatus\tdescription\n')
    for row in mixed:
        stream.write(f"{settings['registration_commit'][:7]}\t{row['oracle_mse']:.10f}\t{peak:.3f}\tkeep\tunchanged {row['family']} data-law mechanism control; no architecture nomination\n")
shutil.copyfile(root/'preflight.json',out/'preflight.json');shutil.copyfile(root/'results.tsv',out/'results.tsv')
print(dict(status='AUDIT_PASS',decision=decision,elapsed=time.monotonic()-started),flush=True)
