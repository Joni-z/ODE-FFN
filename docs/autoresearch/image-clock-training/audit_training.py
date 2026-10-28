"""Boundary-update, native stream, matrix and full inference audits."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
import training_controls as study

def same(a,b):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def update(model,opt,batch):
    opt.zero_grad(set_to_none=True)
    prediction=model(batch['x'],batch['t'],torch.zeros(64,dtype=torch.long))
    residual=prediction-batch['target'];denominator=(1-batch['t'][:,None,None,None]).clamp_min(.05)
    loss=(residual/denominator).square().mean();assert torch.isfinite(loss)
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
    return float(loss.detach()),float(norm)
def training(root,out):
    assert study.load(root/'output/status.json')['state']=='COMPLETE';images,ids=study.old().data_for(study.PARENTS['native']);b=study.bank();rows=[]
    for arm in study.ARMS:
        folder=root/'output/runs'/arm;records=[json.loads(v) for v in (folder/'metrics.jsonl').read_text().splitlines()]
        streams=[json.loads(v) for v in (folder/'streams.jsonl').read_text().splitlines()]
        assert len(records)==7 and len(streams)==21
        for v in streams:assert v['digest']==study.native_state(v['step'])['input_stream_digest']
        for step in study.STEPS:
            post=torch.load(folder/f'step_{step:07d}.pt',weights_only=False,map_location='cpu');assert post['step']==step
            assert post['input_stream_digest']==study.native_state(step)['input_stream_digest']
            model,opt,rng,digest=study.restored(arm,post,independent=True);model.eval();prediction=study.old().predict(model,b)
            with np.load(folder/f'validation_{step:07d}.npz') as f:same(prediction,f['prediction'])
            delta=b['delta'].astype(np.float64)[:,None,None,None];error=prediction.astype(np.float64)-b['target'].astype(np.float64)
            record=next(v for v in records if v['step']==step)
            np.testing.assert_allclose([record['weighted_mse'],record['clean_mse']],[np.mean((error/delta)**2),np.mean(error**2)],rtol=1e-12,atol=1e-12)
            for t in (.1,.3,.5,.7,.9):
                value=study.old().predict(model,study.bank(t))
                with np.load(folder/f'panel_{step:07d}_{t:.1f}.npz') as f:same(value,f['prediction'])
            actual=study.matrices(model,arm)
            with np.load(folder/f'matrices-{step:07d}.npz') as f:
                assert set(f.files)==set(actual)
                for name,value in actual.items():same(value,f[name])
            history=study.load(folder/f'matrix-history-{step:07d}.json');assert len(history)==12
            for row in history:np.testing.assert_allclose(np.linalg.svd(actual[row['layer']].astype(np.float64),compute_uv=False),row['singular_values'],rtol=1e-12,atol=1e-12)
            if arm=='fixed06':
                initial=study.native_state(0)
                for name,value in post['model'].items():
                    if '.t_func.' in name:assert torch.equal(value,initial['model'][name])
            if step:
                pre=torch.load(folder/f'before_{step:07d}.pt',weights_only=False,map_location='cpu');assert pre['step']==step-1
                model,opt,rng,digest=study.restored(arm,pre,independent=True);model.train();batch=study.old().draw(images,ids,rng)
                digest=study.old().fingerprint(digest,batch);update(model,opt,batch);model.eval();study.old().predict(model,b)
                observed=study.old().checkpoint(model,opt,rng,step,digest);study.old().equal(observed,post)
            rows.append(dict(arm=arm,step=step,validation_predictions_bitwise=1024,time_panels_bitwise=5,
                matrix_arrays_bitwise=12,boundary_update_bitwise=bool(step),native_stream_equal=True))
            study.write(out/'progress.json',dict(completed=len(rows),total=14));print(json.dumps(rows[-1]),flush=True)
    return dict(rows=rows,all_native_stream_digests=42,physical_boundary_updates=12,validation_prediction_rows=14336,
        full_optimization_trajectory_replayed=False)
def manual(model,source,progress=None):
    values=[]
    with torch.no_grad():
        for start in range(0,len(source),64):
            x=torch.from_numpy(np.array(source[start:start+64]));labels=torch.zeros(len(x),dtype=torch.long)
            for j in range(128):
                t=torch.full((len(x),),j/128);nt=torch.full_like(t,(j+1)/128)
                first=(model(x,t,labels)-x)/(1-t[:,None,None,None]).clamp_min(.05)
                trial=x+first/128;second=(model(trial,nt,labels)-trial)/(1-nt[:,None,None,None]).clamp_min(.05)
                x=x+(first+second)/256
            values.append(x.numpy())
            if progress:progress(start+len(x))
    return np.concatenate(values)
def sampling(root,out):
    rows=[]
    for arm,step in study.SAMPLES:
        model=study.sample_model(root,arm,step,independent=True);before={k:v.detach().clone() for k,v in model.state_dict().items()};buffers={k:v.detach().clone() for k,v in model.named_buffers()}
        if arm=='native_posthoc06':
            prediction=study.old().predict(model,study.bank())
            with np.load(root/'sampling/posthoc-validation.npz') as f:same(prediction,f['prediction'])
            for t in (.1,.3,.5,.7,.9):
                pred=study.old().predict(model,study.bank(t))
                with np.load(root/'sampling'/f'posthoc-panel-{t:.1f}.npz') as f:same(pred,f['prediction'])
        def progress(count):study.write(out/'progress.json',dict(completed=len(rows),total=7,arm=arm,step=step,images=count))
        observed=manual(model,np.load(study.PARENTS['native']/'banks/sampling-source.npy'),progress)
        same(observed,np.load(root/'sampling'/f'{arm}-{step:07d}.npy'))
        for name,value in model.state_dict().items():assert torch.equal(value,before[name])
        for name,value in model.named_buffers():assert torch.equal(value,buffers[name])
        rows.append(dict(arm=arm,step=step,generated_images_bitwise=1024,parameters_buffers_unchanged=True));print(json.dumps(rows[-1]),flush=True)
    return dict(rows=rows,new_generated_images_bitwise=7168)
def fid(a,b):
    import scipy.linalg
    a=a.astype(np.float64);b=b.astype(np.float64);x=a-a.mean(0);y=b-b.mean(0)
    s=scipy.linalg.svd(x@y.T,compute_uv=False,lapack_driver='gesvd')
    return float(np.sum((a.mean(0)-b.mean(0))**2)+np.sum(x*x)/(len(a)-1)+np.sum(y*y)/(len(b)-1)-2*s.sum()/np.sqrt((len(a)-1)*(len(b)-1)))
def features(root,out):
    sys.path.insert(0,str(study.PARENTS['native']/'source'));import image_metrics as m
    from metric_core import uint8_images
    extractor=m.extractor(study.PARENTS['native']);ref=m.features(extractor,np.load(study.PARENTS['native']/'banks/fid-reference.uint8.npy'))
    same(ref,np.load(study.PARENTS['native']/'metric-inputs/reference-features.npy'));rows=[]
    for row in study.load(root/'metric-output/result.json')['rows']:
        arm,step=row['arm'],row['step'];stored=np.load(study.feature_path(root,arm,step));new=arm in study.ARMS or arm=='native_posthoc06'
        if new:same(m.features(extractor,uint8_images(np.load(root/'sampling'/f'{arm}-{step:07d}.npy'))),stored)
        value=fid(stored,ref);assert abs(value-row['fid'])<1e-8
        rows.append(dict(arm=arm,step=step,fid=value,difference=value-row['fid'],new_features_bitwise=new));print(json.dumps(rows[-1]),flush=True)
    assert len(rows)==32
    return dict(rows=rows,new_generated_features_bitwise=7168,reference_features_bitwise=2048)
def main(root,phase):
    torch.set_num_threads(1);study.verify(root);out=root/f'audit-{phase}';out.mkdir();tick=time.monotonic()
    original={str(p):study.sha(p) for name in ('output','sampling','metric-output') for p in (root/name).rglob('*') if p.is_file()}
    result=globals()[phase](root,out)
    for p,digest in original.items():assert study.sha(p)==digest
    study.verify(root);study.write(out/'result.json',dict(status='PASS',phase=phase,**result,seconds=time.monotonic()-tick,formal_files_unchanged=len(original)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=('training','sampling','features'),required=True);a=p.parse_args();main(a.root,a.phase)
