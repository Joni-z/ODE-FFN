"""Independent stream/spectrum reconstruction and native endpoint replay."""
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
from torch.nn import functional as F

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=args.root.resolve();out=args.output.resolve();out.mkdir(exist_ok=False);sys.path.insert(0,str(root/'source'))
import study
settings,native,_,_=study.setup(root);started=time.monotonic();study.verify_inputs(root,native)
assert study.load(root/'output/status.json')['state']=='COMPLETE'
formal={str(path):native.sha(path) for path in (root/'output').rglob('*') if path.is_file()}
counts=dict(checkpoints=0,panels=0,sampling=0,stream_batches=0,transport_matrices=0,conditional_generators=0,historical_streams=0)
comparison=[];curves=[];training=[];matrix_rows=[];spectra=[];time_panels=[];cost_rows=[]


def close(a,b):np.testing.assert_allclose(a,b,rtol=2e-6,atol=1e-9)


def tensor_equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:tensor_equal(a[key],b[key])
    else:assert a==b


for entry in settings['laws']:
    law=entry['law'];banks=Path(entry['banks']);bank=study.training_bank(entry,native)
    generator=torch.Generator().manual_seed(100000);digest=native.EMPTY
    digests={0:digest};rng_states={0:generator.get_state().clone()}
    for step in range(1,5001):
        target=bank[torch.randint(len(bank),(512,),generator=generator)]
        source=torch.randn(target.shape,generator=generator);t=torch.rand((512,1),generator=generator)
        digest=native.legacy.fingerprint_batch(digest,target,source,t)
        if step in study.STEPS:digests[step]=digest;rng_states[step]=generator.get_state().clone()
    counts['stream_batches']+=5000
    for family,folder in entry['references'].items():
        for step in study.STEPS:
            old=torch.load(Path(folder)/f'step_{step:07d}.pt',weights_only=False)
            assert old['input_stream_digest']==digests[step],(law,family,step)
            assert torch.equal(old['data_rng'],rng_states[step])
        counts['historical_streams']+=1
    with np.load(banks/'field.npz') as data:field_x,field_t=data['x'],data['t']
    assert len(field_x)==(10000 if entry['dim']==2 else 20000)
    for arm in study.ARMS:
        folder=root/'output/runs'/law/arm;assert study.load(folder/'status.json')['state']=='COMPLETE'
        records=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        for step in study.STEPS:
            model=study.build(root,law,arm);state=torch.load(folder/f'step_{step:07d}.pt',weights_only=False)
            native.restore_model(model,state);assert state['input_stream_digest']==digests[step]
            assert torch.equal(state['data_rng'],rng_states[step])
            # Verify finite, correctly shaped Adam moments and the actual update counter.
            params=list(model.parameters());ids=state['optimizer']['param_groups'][0]['params']
            assert len(params)==len(ids)
            if step:
                for parameter,index in zip(params,ids):
                    moment=state['optimizer']['state'][index]
                    assert int(moment['step'])==step
                    for key in ('exp_avg','exp_avg_sq'):
                        assert moment[key].shape==parameter.shape and torch.isfinite(moment[key]).all()
            else:assert not state['optimizer']['state']
            metrics=native.legacy.field_evaluation(model,banks)
            record=next(r for r in records if r['step']==step)
            for key in ('oracle_mse','pair_mse','prediction_rms'):close(metrics[key],record[key])
            with torch.no_grad():prediction=np.concatenate([model(torch.from_numpy(field_x[k:k+1024]),torch.from_numpy(field_t[k:k+1024])).numpy() for k in range(0,len(field_x),1024)])
            with np.load(folder/f'field_{step:07d}.npz') as data:np.testing.assert_array_equal(prediction,data['prediction'])
            curves.append(dict(law=law,arm=arm,step=step,**metrics));counts['checkpoints']+=1
            if arm=='transport':
                visits=[];hooks=[]
                for index,block in enumerate(model.raw.blocks):
                    hooks.append(block.ffn.register_forward_pre_hook(lambda module,args,index=index:visits.append((index,module,args[0].detach().clone()))))
                with np.load(banks/'probe.npz') as data:
                    with torch.no_grad():model(torch.from_numpy(data['x'][:32]),torch.from_numpy(data['t'][:32]))
                for hook in hooks:hook.remove()
                saved=study.load(folder/f'generator_{step:07d}.json');assert len(saved)==len(visits)==3
                for (index,module,x),old in zip(visits,saved):
                    c=module.transport.detach().double().numpy();singular=np.linalg.svd(c,compute_uv=False)
                    np.testing.assert_allclose(singular,old['rectangular_C_singular_values'],rtol=1e-9,atol=1e-10)
                    with torch.no_grad():
                        activation,value=module.w12(x).chunk(2,-1);gate=F.silu(activation)
                        saved_context_values=module.w12(x[:2]).chunk(2,-1)[1]
                        moved=.5*F.linear(gate[:,module.left_width:],module.transport)
                        delta=torch.cat([moved,torch.zeros_like(gate[:,module.left_width:])],-1)
                        ff_delta=F.linear(delta*value,module.w3.weight)
                        ordinary=module.w3(gate*value)
                    ratio=float(delta.square().mean().sqrt()/gate.square().mean().sqrt().clamp_min(1e-12))
                    output_ratio=float(ff_delta.square().mean().sqrt()/ordinary.square().mean().sqrt().clamp_min(1e-12))
                    r=module.w3.in_features;m=module.left_width;p=np.zeros((r,r));p[:m,m:]=c
                    assert np.array_equal(p@p,np.zeros_like(p))
                    for vv,expected in zip(saved_context_values.detach().double().numpy(),old['conditional_K_singular_values']):
                        k=np.zeros((2*r,2*r));k[:r,:r]=p;k[r:,:r]=np.diag(vv)
                        assert np.array_equal(k@k@k,np.zeros_like(k))
                        np.testing.assert_allclose(np.linalg.svd(k,compute_uv=False),expected,rtol=1e-8,atol=1e-9)
                        counts['conditional_generators']+=1
                    matrix_rows.append(dict(law=law,step=step,layer=index,C_frobenius=float(np.linalg.norm(c)),
                        C_sigma_max=float(singular[0]),C_sigma_min=float(singular[-1]),
                        gate_correction_rms_over_gate=ratio,local_FFN_correction_rms_over_ordinary=output_ratio,
                        P_square_zero=True,K_cube_zero=True))
                    for j,sigma in enumerate(singular):spectra.append(dict(law=law,step=step,layer=index,index=j,C_singular_value=float(sigma)))
                    counts['transport_matrices']+=1
        result=study.load(folder/'result.json');assert result['input_stream_digest']==digests[5000]
        for key in ('oracle_mse','pair_mse','prediction_rms'):close(metrics[key],result[key])
        measured=native.compute_budget(model)
        for key in ('parameters','dense_forward_macs','dense_training_forward_backward_macs'):assert measured[key]==result[key]
        cost_rows.append(dict(law=law,arm=arm,**measured))
        for clock in study.TIMES:
            with np.load(banks/f'panel_{clock:.1f}.npz') as data:x,t,truth,jj=[data[k] for k in ('x','t','oracle','oracle_jacobian')]
            value,jac=native.legacy.jacobian(model,torch.from_numpy(x),torch.from_numpy(t))
            with np.load(folder/'diagnostics'/f'panel_{clock:.1f}.npz') as data:
                np.testing.assert_array_equal(value.numpy(),data['prediction']);np.testing.assert_array_equal(jac.numpy(),data['jacobian'])
            old=next(r for r in result['panels'] if abs(r['time']-clock)<1e-6)
            close(float(np.mean((value.numpy()-truth)**2)),old['field_mse'])
            close(float(np.mean(np.sum((jac.numpy()-jj)**2,axis=(1,2)))),old['jacobian_error'])
            time_panels.append(dict(law=law,arm=arm,**old));counts['panels']+=1
        with np.load(banks/'sampling.npz') as data:source,target,directions=[torch.from_numpy(data[k]) for k in ('source','target','directions')]
        generated=native.legacy.generated_samples(model,source,32)
        with np.load(folder/'diagnostics/samples.npz') as data:np.testing.assert_array_equal(generated.numpy(),data['generated'])
        close(native.legacy.sliced_w2(generated,target,directions),result['sw2']);counts['sampling']+=1
        comparison.append(dict(law=law,arm=arm,reused=False,**{key:result[key] for key in ('oracle_mse','jacobian_error','sw2','sw2_reference_floor','parameters','dense_forward_macs','dense_training_forward_backward_macs')}))
        for row in records:training.append(dict(law=law,arm=arm,**row))
        print(dict(law=law,arm=arm,audited=counts,elapsed=time.monotonic()-started),flush=True)
    for family,folder in entry['references'].items():
        result=study.load(Path(folder)/'result.json')
        model=study.build(root,law,family);native.restore_model(model,torch.load(Path(folder)/'step_0005000.pt',weights_only=False))
        metrics=native.legacy.field_evaluation(model,banks)
        for key in ('oracle_mse','pair_mse','prediction_rms'):close(metrics[key],result[key])
        comparison.append(dict(law=law,arm=family,reused=True,**{key:result[key] for key in ('oracle_mse','jacobian_error','sw2','sw2_reference_floor','parameters','dense_forward_macs','dense_training_forward_backward_macs')}))

assert counts==dict(checkpoints=60,panels=75,sampling=15,stream_batches=25000,transport_matrices=60,conditional_generators=120,historical_streams=10),counts
study.verify_inputs(root,native)
for path,digest in formal.items():assert native.sha(path)==digest,path


def table(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (out/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


ratios=[];gates=[]
for entry in settings['laws']:
    law=entry['law'];group=[r for r in comparison if r['law']==law];candidate=next(r for r in group if r['arm']=='transport')
    generic=[r for r in group if r['arm']!='transport'];reference=next(r for r in generic if r['arm']=='swiglu143')
    ratio={key:candidate[key]/min(r[key] for r in generic) for key in ('oracle_mse','jacobian_error','sw2')}
    ratios.append(dict(law=law,**ratio))
    gates.append(dict(law=law,field_within5percent=ratio['oracle_mse']<=1.05,
        gain10percent_on16D=entry['dim']==16 and ratio['oracle_mse']<=.9,
        jacobian_within10percent=law=='gaussian_aniso_2d' or ratio['jacobian_error']<=1.10,
        sampling_within10percent=law=='gaussian_aniso_2d' or ratio['sw2']<=1.10,
        inference_budget=candidate['dense_forward_macs']<=1.05*reference['dense_forward_macs']))
passed=any(g['gain10percent_on16D'] for g in gates) and all(g['field_within5percent'] and g['jacobian_within10percent'] and g['sampling_within10percent'] and g['inference_budget'] for g in gates)
for name,rows in [('comparison.csv',comparison),('curves.csv',curves),('training.csv',training),('matrix-trajectories.csv',matrix_rows),('matrix-spectra.csv',spectra),('time-panels.csv',time_panels),('costs.csv',cost_rows),('frontier-ratios.csv',ratios)]:table(name,rows)
study.write(out/'decision.json',dict(status='nominate_for_frozen_continuation' if passed else 'stop_this_setting',gates=gates,
    no_partition_coefficient_rank_or_feedback_rescue=True,quality_claim_accepted=False))
study.write(out/'audit.json',dict(status='PASS',counts=counts,bindings=len(study.load(root/'bindings.json')),formal_files_unchanged=len(formal),elapsed_seconds=time.monotonic()-started))
telemetry=[json.loads(line) for line in (root/'output/monitor.jsonl').read_text().splitlines()]
peak=max(sum(p['rss_bytes'] for p in row['processes']) for row in telemetry)/2**30
study.write(out/'execution.json',dict(formal=study.load(root/'output/status.json'),new_training_runs=15,reused_endpoints=10,
    model_seed=0,peak_sampled_summed_rss_gib=peak,monitor_samples=len(telemetry),registration_commit=settings['registration_commit']))
shutil.copyfile(root/'preflight.json',out/'preflight.json')
with (root/'results.tsv').open('w') as stream:
    stream.write('commit\tmetric\tmemory_gb\tstatus\tdescription\n')
    for row in comparison:
        status=('keep' if passed else 'discard') if row['arm']=='transport' else 'keep'
        stream.write(f"{settings['registration_commit'][:7]}\t{row['oracle_mse']:.10f}\t{peak:.3f}\t{status}\t{row['law']} {row['arm']} reused={row['reused']}\n")
shutil.copyfile(root/'results.tsv',out/'results.tsv')
print(dict(status='AUDIT_PASS',candidate_passed=passed,ratios=ratios,elapsed=time.monotonic()-started),flush=True)
