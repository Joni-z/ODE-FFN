"""Replay every new head/context endpoint and reconstruct streams and P histories."""
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
from torch import nn

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=args.root.resolve();out=args.output.resolve();out.mkdir(exist_ok=False);sys.path.insert(0,str(root/'source'))
import study
settings,native,core=study.setup(root);started=time.monotonic();study.verify_inputs(root,native)
assert study.load(root/'output/status.json')['state']=='COMPLETE'
from verify_native_heads import verify as verify_native_heads
study.write(out/'native-head-verification.json',verify_native_heads(root,native,core))
formal={str(p):native.sha(p) for p in (root/'output').rglob('*') if p.is_file()};basis=np.load(root/'basis.npy')
counts=dict(checkpoints=0,panels=0,sampling=0,stream_batches=0,P_matrices=0,oracle_samplers=0)
comparison=[];curves=[];training=[];panels=[];matrices=[];singular_rows=[];embedding_rows=[];clock_rows=[]


def close(a,b):np.testing.assert_allclose(a,b,rtol=2e-6,atol=1e-9)


class ReplayHead(nn.Module):
    def __init__(self,raw,head):
        super().__init__();self.raw=raw;self.head=head;self.dim=128;self.mode='native'
    def forward(self,x,t,**options):
        value=self.raw(x,t);denominator=torch.maximum(1-t,torch.full_like(t,.05))
        if self.head=='clean':return (value-x)/denominator
        if self.head=='shifted':return value-x/denominator
        return value
    def ode_layers(self):yield from self.raw.ode_layers()


def field_metrics(prediction,bank):
    error=prediction.astype(np.float64)-bank['oracle'];tangent=error@basis;normal=error-tangent@basis.T
    whole=float(np.mean(error**2));tangent_mse=float(np.mean(tangent**2));normal_mse=float(np.sum(normal**2)/(len(error)*126))
    np.testing.assert_allclose(whole,(2*tangent_mse+126*normal_mse)/128,rtol=1e-10,atol=1e-12)
    decoded=bank['x']+np.maximum(1-bank['t'],.05)*prediction.astype(np.float64)
    return dict(oracle_mse=whole,pair_mse=float(np.mean((prediction-bank['pair'])**2)),prediction_rms=float(np.sqrt(np.mean(prediction**2))),
        tangent_mse_per_dimension=tangent_mse,normal_mse_per_dimension=normal_mse,clean_oracle_mse=float(np.mean((decoded-bank['clean'])**2)))


# Independently regenerate the paired target arrays and input/time stream.
rng=lambda role:np.random.default_rng(np.random.SeedSequence([0,20261028,role]))
regenerated_train=core.paired_targets(rng(702),65536,basis);regenerated_validation=core.paired_targets(rng(703),20000,basis)
random=rng(707);field_t=random.random((20000,1)).astype(np.float32);field_source=random.standard_normal((20000,128)).astype(np.float32)
cold={}
for law_index,entry in enumerate(settings['laws']):
    law=entry['law'];banks=Path(entry['banks']);bank=study.training_bank(entry,native)
    np.testing.assert_array_equal(bank.numpy(),regenerated_train[law_index])
    with np.load(banks/'field.npz') as data:field={k:data[k] for k in data.files}
    np.testing.assert_array_equal(field_t,field['t'])
    np.testing.assert_array_equal((1-field_t)*field_source+field_t*regenerated_validation[law_index],field['x'])
    np.testing.assert_array_equal((1-field_t)/np.maximum(1-field_t,.05)*(regenerated_validation[law_index]-field_source),field['pair'])
    generator=torch.Generator().manual_seed(100000);digest=native.EMPTY;digests={0:digest};rng_states={0:generator.get_state().clone()}
    for step in range(1,5001):
        target=bank[torch.randint(len(bank),(512,),generator=generator)];source=torch.randn(target.shape,generator=generator);t=torch.rand((512,1),generator=generator)
        digest=native.legacy.fingerprint_batch(digest,target,source,t)
        if step in study.STEPS:digests[step]=digest;rng_states[step]=generator.get_state().clone()
    counts['stream_batches']+=5000
    with np.load(banks/'sampling.npz') as data:source,target,target2,directions=[torch.from_numpy(data[k]) for k in ('source','target','target2','directions')]
    class Oracle:
        def __call__(self,x,t):return torch.from_numpy(core.oracle(x.numpy(),t.numpy(),basis,entry['normal_variance'])['oracle']).to(x.dtype)
    oracle_samples=native.legacy.generated_samples(Oracle(),source,32)
    with np.load(banks/'oracle-samples.npz') as data:np.testing.assert_array_equal(oracle_samples.numpy(),data['generated'])
    close(native.legacy.sliced_w2(oracle_samples,target,directions),study.load(banks/'oracle-sampling.json')['sw2']);counts['oracle_samplers']+=1
    for arm in study.ARMS:
        family,head=arm.rsplit('_',1);folder=root/'output/runs'/law/arm;assert study.load(folder/'status.json')['state']=='COMPLETE'
        records=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        model=ReplayHead(native.old_build(128,family,'direct').raw,head)
        for step in study.STEPS:
            state=torch.load(folder/f'step_{step:07d}.pt',weights_only=False);native.restore_model(model,state);model.eval()
            assert state['input_stream_digest']==digests[step];assert torch.equal(state['data_rng'],rng_states[step])
            if step==0:
                if family not in cold:cold[family]=state
                for key,value in cold[family]['model'].items():assert torch.equal(value,state['model'][key])
                for key,value in cold[family]['runtime_buffers'].items():assert torch.equal(value,state['runtime_buffers'][key])
                assert torch.equal(cold[family]['torch_rng'],state['torch_rng'])
                assert not state['optimizer']['state']
            else:
                params=list(model.parameters());ids=state['optimizer']['param_groups'][0]['params'];assert len(params)==len(ids)
                for parameter,index in zip(params,ids):
                    moment=state['optimizer']['state'][index];assert int(moment['step'])==step
                    for key in ('exp_avg','exp_avg_sq'):assert moment[key].shape==parameter.shape and torch.isfinite(moment[key]).all()
            with torch.no_grad():prediction=np.concatenate([model(torch.from_numpy(field['x'][k:k+1024]),torch.from_numpy(field['t'][k:k+1024])).numpy() for k in range(0,len(field['x']),1024)])
            with np.load(folder/f'field_{step:07d}.npz') as data:np.testing.assert_array_equal(prediction,data['prediction'])
            metrics=field_metrics(prediction,field);record=next(r for r in records if r['step']==step)
            for key,value in metrics.items():close(value,record[key])
            curves.append(dict(law=law,family=family,head=head,step=step,**metrics));counts['checkpoints']+=1
            # Descriptive checkpoint geometry; this is not a continuation criterion.
            a=model.raw.input.weight.detach().double().numpy();_,sigma,vh=np.linalg.svd(a,full_matrices=False);row=vh[sigma>1e-10]
            capture=float(np.sum((row@basis)**2)/2)
            embedding_rows.append(dict(law=law,family=family,head=head,step=step,input_rank=len(row),latent_plane_capture=capture))
            saved=study.load(folder/f'generator_{step:07d}.json');layers=list(model.ode_layers());assert len(saved)==len(layers)
            for old,(name,layer) in zip(saved,layers):
                assert old['layer']==name;p=layer.proj.weight.detach().double();sv=torch.linalg.svdvals(p).numpy()
                np.testing.assert_allclose(sv,old['P_singular_values'],rtol=1e-8,atol=1e-9)
                ev=np.linalg.eigvals(p.numpy());np.testing.assert_allclose(ev.real,old['P_eigenvalues_real'],rtol=1e-8,atol=1e-9)
                np.testing.assert_allclose(ev.imag,old['P_eigenvalues_imag'],rtol=1e-8,atol=1e-9)
                matrices.append(dict(law=law,head=head,step=step,layer=name,sigma_max=float(sv[0]),sigma_min=float(sv[-1]),
                    sigma_mean=float(sv.mean()),sigma_cv=float(sv.std()/sv.mean()),spectral_radius=float(np.max(np.abs(ev))),
                    nonnormal_commutator_ratio=float(torch.linalg.norm(p.T@p-p@p.T)/(torch.linalg.norm(p)**2))))
                for index,value in enumerate(sv):singular_rows.append(dict(law=law,head=head,step=step,layer=name,index=index,singular_value=float(value)))
                counts['P_matrices']+=1
            clock_saved=study.load(folder/f'clocks_{step:07d}.json');clock_now=native.legacy.visit_diagnostics(model,banks)
            assert clock_now==clock_saved
            for visit in clock_now:clock_rows.append(dict(law=law,head=head,step=step,layer=visit['layer'],visit=visit['visit'],**visit['activation']))
        result=study.load(folder/'result.json')
        for key,value in metrics.items():close(value,result[key])
        assert result['input_stream_digest']==digests[5000]
        measured=native.compute_budget(model)
        for key in ('parameters','dense_forward_macs','dense_training_forward_backward_macs'):assert measured[key]==result[key]
        for clock in study.TIMES:
            with np.load(banks/f'panel_{clock:.1f}.npz') as data:x,t,truth,jj=[data[k] for k in ('x','t','oracle','oracle_jacobian')]
            value,jac=native.legacy.jacobian(model,torch.from_numpy(x),torch.from_numpy(t))
            with np.load(folder/'diagnostics'/f'panel_{clock:.1f}.npz') as data:
                np.testing.assert_array_equal(value.numpy(),data['prediction']);np.testing.assert_array_equal(jac.numpy(),data['jacobian'])
            old=next(r for r in result['panels'] if abs(r['time']-clock)<1e-6)
            close(float(np.mean((value.numpy()-truth)**2)),old['field_mse'])
            actual=float(np.mean(np.sum((jac.numpy()-jj)**2,axis=(1,2))));close(actual,old['jacobian_error'])
            bounds=core.bottleneck_bounds(t,entry['normal_variance']);bound=float(np.mean(bounds['direct_J_floor' if head=='direct' else 'skip_J_floor']))
            assert actual>=bound*(1-1e-5)-1e-7,(law,arm,clock,actual,bound)
            panels.append(dict(law=law,family=family,head=head,time=clock,**{k:v for k,v in old.items() if k!='time'},
                architecture_J_floor=bound));counts['panels']+=1
        generated=native.legacy.generated_samples(model,source,32)
        with np.load(folder/'diagnostics/samples.npz') as data:np.testing.assert_array_equal(generated.numpy(),data['generated'])
        close(native.legacy.sliced_w2(generated,target,directions),result['sw2']);counts['sampling']+=1
        angles=torch.arange(128)*2*torch.pi/128;td=torch.stack((torch.cos(angles),torch.sin(angles)),-1)
        close(native.legacy.sliced_w2(torch.tensor(generated.numpy()@basis,dtype=torch.float32),torch.tensor(target.numpy()@basis,dtype=torch.float32),td),result['tangent_sw2'])
        values={k:v for k,v in result.items() if isinstance(v,(float,int))}
        comparison.append(dict(law=law,family=family,head=head,**values))
        for row in records:training.append(dict(law=law,family=family,head=head,**row))
        print(dict(law=law,arm=arm,counts=counts,elapsed=time.monotonic()-started),flush=True)

assert counts==dict(checkpoints=72,panels=90,sampling=18,stream_batches=10000,P_matrices=144,oracle_samplers=2),counts
study.verify_inputs(root,native)
for path,digest in formal.items():assert native.sha(Path(path))==digest,path


def table(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (out/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


interactions=[]
for head in ('clean','shifted'):
    checks=[]
    for law in ('thin','thick'):
        group=[r for r in comparison if r['law']==law and r['head']==head];ode=next(r for r in group if r['family']=='ode_k2');generic=[r for r in group if r['family']!='ode_k2']
        ratios={key:ode[key]/min(r[key] for r in generic) for key in ('oracle_mse','tangent_mse_per_dimension','jacobian_error','sw2')}
        checks.append(dict(law=law,**ratios))
    thin,thick=checks
    eligible=thin['oracle_mse']<=.9 and thick['oracle_mse']<=1.05 and all(r[key]<=1.1 for r in checks for key in ('tangent_mse_per_dimension','jacobian_error','sw2'))
    interactions.append(dict(head=head,context_specific_interaction=eligible,checks=checks))
study.write(out/'decision.json',dict(status='compatibility_check_complete',new_architecture_nominated=False,
    context_specific_interaction_found=any(r['context_specific_interaction'] for r in interactions),interactions=interactions,
    generic_head_effect_not_an_ODE_claim=True,no_head_cap_sampler_basis_dimension_or_width_rescue=True))
for name,rows in [('comparison.csv',comparison),('curves.csv',curves),('training.csv',training),('time-panels.csv',panels),('P-trajectories.csv',matrices),('P-singular-values.csv',singular_rows),('input-geometry.csv',embedding_rows),('clocks.csv',clock_rows)]:table(name,rows)
telemetry=[json.loads(line) for line in (root/'output/monitor.jsonl').read_text().splitlines()];peak=max(sum(p['rss_bytes'] for p in row['processes']) for row in telemetry)/2**30
study.write(out/'audit.json',dict(status='PASS',counts=counts,bindings=len(study.load(root/'bindings.json')),formal_files_unchanged=len(formal),elapsed_seconds=time.monotonic()-started))
study.write(out/'execution.json',dict(formal=study.load(root/'output/status.json'),new_training_runs=18,peak_sampled_summed_rss_gib=peak,model_seed=0,
    registration_commit=settings['registration_commit'],monitor_samples=len(telemetry)))
shutil.copyfile(root/'preflight.json',out/'preflight.json')
with (root/'results.tsv').open('w') as stream:
    stream.write('commit\tmetric\tmemory_gb\tstatus\tdescription\n')
    for row in comparison:stream.write(f"{settings['registration_commit'][:7]}\t{row['oracle_mse']:.10f}\t{peak:.3f}\tkeep\tcompatibility only {row['law']} {row['family']} {row['head']}; no nomination\n")
shutil.copyfile(root/'results.tsv',out/'results.tsv');print(dict(status='AUDIT_PASS',counts=counts,elapsed=time.monotonic()-started),flush=True)
