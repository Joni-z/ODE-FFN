"""Independent updates, Heun sampling and feature/FID replays for both arms."""
from __future__ import annotations
import json,time
import numpy as np
import torch
import clock_source_study as study

def helper():
    support,_,_,_=study.parents()
    return study.module('clock_source_pure_audit_helpers',support.COLD/'source/audit_training.py')
def begin(root,phase,kind):
    study.environment(root,kind);study.verify(root);out=root/f'audit-{phase}';out.mkdir()
    files={str(p):study.sha(p) for name in ('output','sampling','metric-output') for p in (root/name).rglob('*') if p.is_file()}
    return out,files,time.monotonic()
def finish(root,out,files,tick,result):
    for p,digest in files.items():assert study.sha(p)==digest,p
    study.verify(root);study.write(out/'result.json',dict(status='PASS',**result,formal_files_unchanged=len(files),seconds=time.monotonic()-tick));print('COLD_CLOCK_SOURCE_AUDIT_PASS',out.name,flush=True)
def audit_training(root):
    out,files,tick=begin(root,'training','training');support,_,n,_=study.parents();assert study.load(root/'output/status.json')['state']=='COMPLETE';pure=helper();images,ids=n.data_for(support.NATIVE);_,_,rng,digest=study.restored(study.ARMS[0]);digests={}
    for step in range(1,20001):
        batch=n.draw(images,ids,rng);digest=n.fingerprint(digest,batch)
        if step%1000==0 or step==2500:
            assert digest==study.native_state(step)['input_stream_digest'];digests[step]=digest
    assert len(digests)==21;rows=[]
    for arm in study.ARMS:
        folder=root/'output/runs'/arm;assert study.load(folder/'status.json')['state']=='COMPLETE';records=[json.loads(v) for v in (folder/'metrics.jsonl').read_text().splitlines()];streams=[json.loads(v) for v in (folder/'streams.jsonl').read_text().splitlines()];assert [v['step'] for v in records]==list(study.STEPS) and len(streams)==21
        for row in streams:assert row['digest']==digests[row['step']]
        for step,record in zip(study.STEPS,records):
            post=torch.load(folder/f'step_{step:07d}.pt',weights_only=False,map_location='cpu');assert post['step']==step and post['input_stream_digest']==study.native_state(step)['input_stream_digest'];model,opt,rng,digest=study.restored(arm,post,independent=True);model.eval();bank=study.bank();prediction=n.predict(model,bank)
            with np.load(folder/f'validation_{step:07d}.npz') as f:pure.same(prediction,f['prediction'])
            error=prediction.astype(np.float64)-bank['target'].astype(np.float64);delta=bank['delta'].astype(np.float64)[:,None,None,None];np.testing.assert_allclose([record['weighted_mse'],record['clean_mse']],[np.mean((error/delta)**2),np.mean(error**2)],rtol=1e-12,atol=1e-12)
            for t in (.1,.3,.5,.7,.9):
                actual=n.predict(model,study.bank(t))
                with np.load(folder/f'panel_{step:07d}_{t:.1f}.npz') as f:pure.same(actual,f['prediction'])
            arrays=study.matrices(model,arm)
            with np.load(folder/f'matrices-{step:07d}.npz') as f:
                assert set(f.files)==set(arrays)
                for name,value in arrays.items():pure.same(value,f[name])
            history=study.load(folder/f'matrix-history-{step:07d}.json');assert len(history)==12
            for row in history:
                value=arrays[row['layer']].astype(np.float64);np.testing.assert_allclose(np.linalg.svd(value,compute_uv=False),row['singular_values'],rtol=1e-12,atol=1e-12);assert float(np.linalg.norm(value))==row['frobenius']
                if '.P' in row['layer']:
                    e=np.linalg.eigvals(value);np.testing.assert_allclose(e.real,row['eigen_real'],rtol=1e-12,atol=1e-12);np.testing.assert_allclose(e.imag,row['eigen_imag'],rtol=1e-12,atol=1e-12)
            observed=study.clock_profiles(model);assert observed==study.load(folder/f'clock-profile-{step:07d}.json')
            if step:
                pre=torch.load(folder/f'before_{step:07d}.pt',weights_only=False,map_location='cpu');assert pre['step']==step-1;active,opt,rng,digest=study.restored(arm,pre,independent=True);active.train();batch=n.draw(images,ids,rng);digest=n.fingerprint(digest,batch);pure.update(active,opt,batch);active.eval();n.predict(active,bank);n.equal(n.checkpoint(active,opt,rng,step,digest),post)
            rows.append(dict(arm=arm,step=step,validation_images_bitwise=1024,panels_bitwise=5,matrix_arrays_bitwise=12,clock_profiles_bitwise=30,boundary_update_bitwise=bool(step)));study.write(out/'progress.json',dict(completed=len(rows),total=14));print(json.dumps(rows[-1]),flush=True)
    finish(root,out,files,tick,dict(rows=rows,draws_reconstructed=20000,stream_boundaries_verified=42,physical_last_updates=12,full_optimizer_trajectory_replayed=False))
def audit_sampling(root):
    out,files,tick=begin(root,'sampling','training');assert study.load(root/'sampling/result.json')['status']=='PASS';support,_,_,_=study.parents();pure=helper();source=np.load(support.NATIVE/'banks/sampling-source.npy');rows=[]
    for arm,step in study.SAMPLES:
        model=study.model_at(root,arm,step,independent=True);before={k:v.detach().clone() for k,v in model.state_dict().items()};buffers={k:v.detach().clone() for k,v in model.named_buffers()}
        def progress(count):study.write(out/'progress.json',dict(completed=len(rows),total=6,arm=arm,step=step,images=count))
        raw=pure.manual(model,source,progress);pure.same(raw,np.load(root/'sampling'/f'{arm}-{step:07d}.npy'))
        for name,value in model.state_dict().items():assert torch.equal(value,before[name])
        for name,value in model.named_buffers():assert torch.equal(value,buffers[name])
        rows.append(dict(arm=arm,step=step,images_bitwise=1024,parameters_buffers_unchanged=True));print(json.dumps(rows[-1]),flush=True)
    finish(root,out,files,tick,dict(rows=rows,generated_images_bitwise=6144))
def audit_metrics(root):
    out,files,tick=begin(root,'metrics','metrics');result=study.load(root/'metric-output/result.json');assert result['status']=='PASS' and result['historical_rows']==study.historical_rows();support,_,_,_=study.parents();m,core=study.metrics_modules(root);pure=helper();ex=m.extractor(support.NATIVE);reference=np.load(support.NATIVE/'metric-inputs/reference-features.npy');pure.same(m.features(ex,np.load(support.NATIVE/'banks/fid-reference.uint8.npy'),progress='audit-reference'),reference);rows=[]
    assert [(r['arm'],r['step']) for r in result['rows']]==list(study.SAMPLES)
    for row in result['rows']:
        arm,step=row['arm'],row['step'];raw=np.load(root/'sampling'/f'{arm}-{step:07d}.npy');images=np.rint((np.maximum(-1,np.minimum(1,raw.astype(np.float32)))+np.float32(1))*np.float32(127.5)).astype(np.uint8);pure.same(images,core.uint8_images(raw));features=np.load(root/'metric-output'/f'{arm}-{step:07d}-features.npy');pure.same(m.features(ex,images,progress=f'audit-{arm}-{step}'),features);value=pure.fid(features,reference);assert abs(value-row['fid'])<1e-6;rows.append(dict(arm=arm,step=step,features_bitwise=1024,independent_FID=value,absolute_error=abs(value-row['fid'])));study.write(out/'progress.json',dict(completed=len(rows),total=6))
    finish(root,out,files,tick,dict(rows=rows,generated_features_bitwise=6144,reference_features_bitwise=2048,historical_endpoints_retained=35))
